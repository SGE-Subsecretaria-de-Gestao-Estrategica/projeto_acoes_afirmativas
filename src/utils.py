#%%
from pdfminer.high_level                    import extract_text
from langchain_text_splitters               import RecursiveCharacterTextSplitter
from langchain.embeddings                   import OpenAIEmbeddings
from langchain.vectorstores.chroma          import Chroma
from langchain_core.documents               import Document
from langchain_openai                       import ChatOpenAI
from langchain_core.prompts                 import ChatPromptTemplate
from langchain_core.output_parsers          import JsonOutputParser
from typing                                 import List, Dict
import re, random, os
from dotenv import load_dotenv

#%%
CHROMA_PATH = "./chroma"
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
#%%
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


def get_embedding():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key= os.getenv("OPEN_API_KEY")
    )

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


def add_to_chroma(documents: list):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding()
    )
    document_ids = [doc.metadata['id'] for doc in documents]
    db.add_documents(documents, ids=document_ids)
    db.persist()


def call_gpt_4o_mini(texto_completo: str) -> Dict:
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
        Você receberá um edital público completo. Extraia APENAS as seguintes informações exatamente como aparecem no texto. Se não estiverem presentes, use "NaN":

        - valor_total: Qual o valor total do edital?
        - cotas_negras: Qual o percentual de cotas para pessoas negras?
        - cotas_indigenas: Qual o percentual de cotas para pessoas indígenas?
        - cotas_pcd: Qual o percentual de cotas para pessoas com deficiência (pcd)?
        - vagas_totais: Quantos projetos/propostas/vagas serão contemplados/disponibilizados/selecionados?

        Responda SOMENTE com um JSON com as seguintes chaves:
        {{
        "valor_total": "...",
        "cotas_negras": "...",
        "cotas_indigenas": "...",
        "cotas_pcd": "...",
        "vagas_totais": "..."
        }}
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

