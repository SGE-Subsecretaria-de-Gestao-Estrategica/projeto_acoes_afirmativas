from pdfminer.high_level                    import extract_text
from langchain_openai                       import ChatOpenAI
from langchain.embeddings                   import OpenAIEmbeddings
from langchain_core.prompts                 import ChatPromptTemplate
from langchain_text_splitters               import RecursiveCharacterTextSplitter
from langchain_core.documents               import Document
from langchain_core.output_parsers          import JsonOutputParser
from langchain.vectorstores.chroma          import Chroma
from dotenv                                 import load_dotenv
from typing                                 import List, Dict
import re, random, os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def pdf_parser(pdf_path: str) -> str:
    """ Extrai os textos do pdf, tranformando um documento em uma string

    Args:
        pdf_path (str): caminho do pdf

    Returns:
        str: string com o conteúdo do pdf
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Erro ao extrair texto do PDF com pdfminer.six: {e}")
        return None


def chunknizer(
        text: str, 
        chunk_size: int = 800, 
        chunk_overlap: int = 200
) -> list: #TODO -> colocar opção de lista
    """ Divide variável de texto em pedaços menores com base no tamanho e sobreposição especificados.
        Usa quebra de linha para definir a pausa da quabra.
        E.g. 800 + caracteres até a próxima quebra de linha.
    
     Args:
        text (str): _description_
        chunk_size (int, optional): _description_. Defaults to 800.
        chunk_overlap (int, optional): _description_. Defaults to 200.

    Returns:
        list: lista de chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=False,
        separators=["\n\n"]
    )
    return text_splitter.split_text(text)


def filter_regex(chunks: list, padrao_regex: re.Pattern) -> list:
    """_summary_

    Args:
        chunks (list): _description_
        padrao_regex (re.Pattern): _description_

    Returns:
        list: _description_
    """
    
    return [chunk for chunk in chunks if padrao_regex.search(chunk.page_content)]


def get_chunk_ids(edital_id: str, uf_edital: str, chunks: list) -> list:
    documents = []
    used_ids = set()  
    for i, chunk_text in enumerate(chunks): 
        while True:
            random_id = f"{i}-{random.randint(0, len(chunks) * 10000)}" 
            if random_id not in used_ids:
                used_ids.add(random_id)
                break

        # Criar um objeto Document do LangChain
        doc = Document(
            page_content=chunk_text, 
            metadata={
                "id": random_id,
                "edital_id": edital_id,
                "uf_edital": uf_edital
            }
        )
        documents.append(doc)
    return documents



def call_gpt_4o_mini(texto_completo: str) -> Dict:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
    Você é um extrator de dados especializado em editais públicos brasileiros.

    Você receberá um trecho de um edital em formato MARKDOWN. 
        
    TAREFA: Extraia as informações abaixo EXATAMENTE como aparecem no documento.

    INSTRUÇÕES CRÍTICAS PARA VALOR_TOTAL:
    - Procure por: "valor global", "dotação orçamentária", "recursos totais", "investimento total"
    - Extraia o valor NUMÉRICO EXATO (ex: "500000.00" ou "R$ 500.000,00")
    - Se houver múltiplos valores, some-os e informe o total
    - Mantenha o formato original (com R$, pontos, vírgulas)
    - Se o valor estiver por extenso, converta para numérico
    - Se houver faixas (ex: "até R$ 50.000"), extraia o valor máximo

    CAMPOS OBRIGATÓRIOS:
    1. valor_total: Valor total destinado ao edital (string com formato monetário)
    2. cotas_negras: Percentual de cotas para pessoas negras (ex: "20%", "20", "vinte por cento")
    3. cotas_indigenas: Percentual de cotas para pessoas indígenas
    4. cotas_pcd: Percentual de cotas para pessoas com deficiência
    5. vagas_totais: Número total de projetos/propostas/vagas/prêmios a serem contemplados

    REGRAS DE PREENCHIMENTO:
    - Use "NaN" apenas se a informação NÃO existir no documento
    - NÃO invente ou estime valores
    - NÃO use "não informado" ou "não consta" - use apenas "NaN"
    - Mantenha números e símbolos originais (%, R$, etc.)
    - Para cotas, aceite formato percentual ou numérico
    - Para vagas_totais, extraia o número exato de contemplados

    FORMATO DE RESPOSTA (JSON puro, sem markdown):
    {{
        "valor_total": "R$ 500.000,00",
        "cotas_negras": "20%",
        "cotas_indigenas": "NaN",
        "cotas_pcd": "5%",
        "vagas_totais": "10"
    }}

    IMPORTANTE: Responda APENAS com o JSON. Não adicione explicações, comentários ou texto adicional.
        """),
        ("human", "{text}")
    ])

    parser = JsonOutputParser()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv(api_key)
    )

    chain = prompt_template | llm | parser
    return chain.invoke({"text": texto_completo})

