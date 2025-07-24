#%%
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from regex import regex_verificar_acoes_afirmativas, regex_cotas_negros
import re

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
        separators=["\n"]
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
    
    return [chunk for chunk in chunks if padrao_regex.search(chunk)]
