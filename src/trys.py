#%%
from utils import pdf_parser, chunknizer, get_embedding, add_to_chroma, get_chunk_ids
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms import Ollama
import json

#%% Parseia o PDF
pdf = "./../data/data_test/TOCANTINS_Edital_TCC_-_Pontos_PNCV_1-1-18.pdf"
texto_teste = "texto = '7. COTAS\n7.1 Ficam garantidas, conforme descrito no Anexo 1, cotas neste edital para:\n\na. pessoas negras (pretas e pardas): 25% (vinte e cinco por cento) das\n\nvagas;\n\nb. pessoas indígenas: 10% (dez por cento) das vagas;\nc. pessoas com deficiência: 5% (cinco por cento) das vagas.\n\n7.2 As cotas serão destinadas às entidades que possuam quadro de dirigentes\nmajoritariamente (cinquenta por cento mais um) composto por pessoas negras,\nindígenas ou com deficiência, ou que tenham pessoas negras, indígenas ou com\ndeficiência na maioria (cinquenta por cento mais um) das posições de liderança\n(coordenação/direção) no projeto cultural.'"

# text_parsed = pdf_parser(pdf)
# %% Chunkeia 
chunks      = chunknizer(text=texto_teste)
documents   = get_chunk_ids(chunks=chunks)
# %%
add_to_chroma(documents=documents)

# %%
questions = [
    "Qual Valor total do edital?",
    "Qual Percentual de cotas para pessoas negras?",
    "Qual Percentual de cotas para pessoas indígenas?",
    "Qual Percentual de cotas para pessoas com deficiência (pcd)?",
    "Quantos projetos serão selecionados?"
]

PROMPT_TEMPLATE = """"
Instruções:
1. Os valores extraídos devem estar exatamente como no texto
2. Se a informação não estiver no texto, use "NÃO ENCONTRADO"

Texto para análise:
{context}

Responda as pergunta:
{question} 
"""
CHROMA_PATH = "./chroma"

def query_rag(query_text: str):
    # Prepara o DB de embeddings.
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Busca no DB os documentos mais relevantes com seus scores.
    # O score é uma medida de distância/similaridade. Valores menores geralmente indicam maior similaridade.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Obtém o score do documento mais relevante (o primeiro na lista de resultados).
    # Se não houver resultados, define o score como None.
    best_score = results[0][1] if results else None

    # Cria o prompt usando o template e o contexto/pergunta.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt          = prompt_template.format(context=context_text, question=query_text)

    # Instancia o modelo Ollama com qwen3.max e temperature=0 para respostas exatas.
    model = Ollama(model="qwen3.max", temperature=0)
    
    # Invoca o modelo para obter a resposta.
    response_text = model.invoke(prompt)

    # Extrai os IDs das fontes (documentos) utilizadas.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Formata e imprime a resposta para visualização no console.
    formatted_response = f"Response: {response_text}\nSources: {sources}\nScore: {best_score}"
    print(formatted_response)
    
    # Retorna a resposta em um formato de dicionário para ser agregado no JSON final.
    return {
        "answer": response_text,
        "score": best_score,
        "sources": sources # Inclui as fontes no retorno para cada pergunta
    }

# %% Bloco de execução principal
if __name__ == "__main__":
    print("--- Iniciando o processamento das perguntas ---")
    
    # Dicionário para armazenar todas as respostas no formato JSON desejado.
    all_responses_json = {}

    for q in questions:
        print(f"\n--- Processando a pergunta: '{q}' ---")
        # Chama query_rag e armazena a resposta formatada.
        response_data = query_rag(q)
        # Adiciona a pergunta como chave e os dados da resposta como valor.
        all_responses_json[q] = response_data
        
    print("\n--- Processamento de todas as perguntas concluído ---")
    
    # Imprime o JSON final com todas as perguntas e suas respostas/scores.
    print("\n--- Respostas em formato JSON ---")
    # Usa indent=4 para uma saída JSON mais legível.
    print(json.dumps(all_responses_json, indent=4, ensure_ascii=False))

# %%
