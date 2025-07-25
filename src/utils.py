
from pdfminer.high_level                    import extract_text
from langchain_text_splitters               import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama  import OllamaEmbeddings
from langchain.vectorstores.chroma          import Chroma
from langchain_core.documents               import Document
import re, random

CHROMA_PATH = "./chroma"


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
    """ Divide variábel de texto em pedaços menores com base no tamanho e sobreposição especificados.
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
    return OllamaEmbeddings(model= "qwen3.max")


def add_to_chroma(documents: list):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding()
    )
    document_ids = [doc.metadata['id'] for doc in documents]
    db.add_documents(documents, ids=document_ids)
    db.persist()


def get_chunk_ids(chunks: list) -> list:
    documents = []
    used_ids = set()  
    for i, chunk_text in enumerate(chunks): 
        while True:
            random_id = f"{i}-{random.randint(0, len(chunks) * 1000)}" 
            if random_id not in used_ids:
                used_ids.add(random_id)
                break

        # Criar um objeto Document do LangChain
        doc = Document(page_content=chunk_text, metadata={"id": random_id})
        documents.append(doc)
    return documents


