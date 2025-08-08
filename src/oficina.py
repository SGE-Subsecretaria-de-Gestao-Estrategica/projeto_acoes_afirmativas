#%%
import pandas as pd
import re, ast
from regex_patterns import regex_categoria
import os
from utils import (
    pdf_parser, 
    chunknizer, 
    get_chunk_ids
)
from tqdm import tqdm
import pdfplumber

# %%
INPUT_PATH_CAPITAIS = "/Users/gabrielribeirobizerril/Documents/GitHub/llm/editai_extractor_llm_based/data/input/capitais"
# %% Carregamento dos dados
def load_data(input_files: str) -> pd.DataFrame:
    INPUT_PATH = input_files
    l = []
    entes = [nome for nome in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH, nome))]
    for ente in entes:
        pasta = INPUT_PATH + f"/{ente}"
        d     = {
            "ente"       : ente, 
            "path_files" : [pasta + f"/{f}" for f in os.listdir(pasta) if f.endswith('.pdf')],
            "files"      : [f for f in os.listdir(pasta) if f.endswith('.pdf')]

        }
        l.append(d)

    return pd.DataFrame(l)

#%%
df_load_data = load_data(input_files=INPUT_PATH_CAPITAIS)
# %%
l_documents = []
l_dict      = []
for idx, row in df_load_data.iterrows():
    ente = row["ente"]
    for idx, path in enumerate(row["path_files"]):
        id_doc      = row["files"][idx]
        texto       = pdf_parser(path)
        chunks      = chunknizer(text=texto)
        documents   = get_chunk_ids(chunks=chunks, edital_id=id_doc, uf_edital=ente)
        l_documents.append(documents)
        d = {
            "uf": ente,
            "path_pdf": path,
            "pdf": id_doc,
            "document": documents
        }
        l_dict.append(d)

# %%
df_dict = pd.DataFrame(l_dict)
df_dict.to_pickle("df_dict_7_8_25.pkl")
# %%
def limpa_texto(texto: str) -> str:
    # Remove caracteres de controle ilegais (exceto \n, \r, \t se desejar)
    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', texto)
    # Remove múltiplos espaços
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# %%
def filtra_chunks(documents: list):
    regex_categorias       = regex_categoria()

    # Filtragem com regex
    chunks_categorias = [doc for doc in documents if regex_categorias.search(doc.page_content)]

    # União sem duplicatas
    chunks_y = list({doc.metadata['id']: doc for doc in (chunks_categorias)}.values())

    # Limpeza dos caracteres indesejados no conteúdo
    for doc in chunks_y:
        doc.page_content = limpa_texto(doc.page_content)

    return chunks_y

#%%
def aplicar_filtragem(df):
    df = df.copy()

    # Aplica a função que filtra os chunks
    df["chunks_relevantes"] = df["document"].apply(filtra_chunks)

    # Junta os textos dos chunks filtrados
    df["texto_completo"] = df["chunks_relevantes"].apply(
        lambda chunks: "\n\n".join(doc.page_content for doc in chunks)
    )

    return df

df_filtrado = aplicar_filtragem(df_dict)

# %%
df_resultado = df_filtrado.loc[
    df_filtrado["chunks_relevantes"].apply(lambda x: isinstance(x, list) and len(x) > 0)
]
#%%
tqdm.pandas()

palavras_chave = [
    r"cotas?",
    r"modalidades?",
    r"linhas?",
    r"categorias?",
    r"reserva[s]?",
    r"ampla\s+concorr[êe]ncia",
    r"comunidades?\s+tradicionais?",
    r"pretos?\s+e\s+pardos?",
    r"quilombolas?",
    r"negros?|negras?",                     # negros / negras
    r"povos?\s+origin[aá]rios?",            # povos originários
    r"ind[ií]genas?",                       # indígenas
    r"pessoas?\s+com\s+defici[eê]ncia",     # pessoas com deficiência
    r"\bpcd\b"                              # PCD (como palavra isolada)
]


# Regex compilado para performance
regex_keywords = re.compile(r"|".join(palavras_chave), flags=re.IGNORECASE)

def extrair_tabelas_boas(path_pdf, min_cols=4, min_rows=2):
    tabelas_boas = []

    try:
        with pdfplumber.open(path_pdf) as pdf:
            for page in pdf.pages:
                tabelas = page.extract_tables()
                for tabela_raw in tabelas:
                    if not tabela_raw or len(tabela_raw) <= 1:
                        continue

                    header = tabela_raw[0]
                    n_cols = len(header)
                    n_rows = len(tabela_raw) - 1

                    if n_cols < min_cols or n_rows < min_rows:
                        continue

                    colunas_ruins = sum(
                        1 for cell in header if not cell or cell.strip() in ["", "-", None]
                    )
                    if colunas_ruins / n_cols >= 0.5:
                        continue

                    # Junta o cabeçalho em uma string única e verifica se alguma palavra-chave aparece
                    header_text = " ".join([str(cell) for cell in header if cell])
                    if not regex_keywords.search(header_text):
                        continue  # ignora se não tem nenhuma keyword

                    tabelas_boas.append(tabela_raw)

    except Exception as e:
        print(f"Erro ao processar {path_pdf}: {e}")
    
    return tabelas_boas if tabelas_boas else None


def extrair_tabelas_pdf(path_pdf, min_cols=4, min_rows=2):
    tabelas_boas = []

    try:
        with pdfplumber.open(path_pdf) as pdf:
            for page in pdf.pages:
                tabelas = page.extract_tables()
                for tabela_raw in tabelas:
                    if not tabela_raw or len(tabela_raw) <= 1:
                        continue

                    header = tabela_raw[0]
                    n_cols = len(header)
                    n_rows = len(tabela_raw) - 1

                    if n_cols < min_cols or n_rows < min_rows:
                        continue

                    # Conta quantas colunas do cabeçalho são vazias ou irrelevantes
                    colunas_ruins = sum(
                        1 for cell in header if not cell or cell.strip() in ["", "-", None]
                    )

                    proporcao_ruim = colunas_ruins / n_cols

                    if proporcao_ruim >= 0.5:
                        continue  # descarta tabela ruim

                    tabelas_boas.append(tabela_raw)

    except Exception as e:
        print(f"Erro ao processar {path_pdf}: {e}")
    
    return tabelas_boas if tabelas_boas else None
# Aplica a função à coluna 'path_pdf'
df_resultado["tabelas_y"] = df_resultado["path_pdf"].progress_apply(extrair_tabelas_boas)
# %%
# Converte cada tabela bruta em DataFrame
def converter_para_df(tabela_raw):
    try:
        df = pd.DataFrame(tabela_raw[1:], columns=tabela_raw[0])
        df.columns = df.columns.str.replace("\n", " ", regex=True)
        return df
    except Exception as e:
        print(f"Erro ao converter tabela: {e}")
        return None

# Explodir em uma nova lista
tabelas_expandidas = []

for _, row in df_resultado.iterrows():
    if row["tabelas_y"]:
        for tabela_raw in row["tabelas_y"]:
            df_tabela = converter_para_df(tabela_raw)
            if df_tabela is not None:
                tabelas_expandidas.append({
                    "path_pdf": row["path_pdf"],
                    "pdf": row["pdf"],
                    "dataframe": df_tabela
                })

df_tabelas_explodidas_n = pd.DataFrame(tabelas_expandidas)
# %%
