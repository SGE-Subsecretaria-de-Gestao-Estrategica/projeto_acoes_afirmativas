# Imports
from utils import (
    pdf_parser, 
    chunknizer, 
    get_embedding, 
    add_to_chroma, 
    get_chunk_ids,
    call_gpt_4o_mini
)
import  regex_patterns
import  os, re
import  pandas as pd
import  pickle

# Contanates
INPUT_PATH_ESTADOS = r"C:\Users\Gabriel\Documents\GitHub\editai_extractor_llm_based\data\input\editais_estados"
INPUT_PATH_CAPITAIS = "/Users/gabrielribeirobizerril/Documents/GitHub/llm/editai_extractor_llm_based/data/input/capitais"
# Carregamento dos dados
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

df_load_data = load_data(input_files=INPUT_PATH_CAPITAIS)


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

# Salva os arquivos já pré-processados
with open("./../data/output/capitais/l_documents_capitais_versao_clean.pkl", "wb") as f:
    pickle.dump(l_documents, f)
    
with open("./../data/output/capitais/l_dict_capitais_versao_clean.pkl", "wb") as f:
    pickle.dump(l_dict, f)

df_dict = pd.DataFrame(l_dict)
df_dict.to_csv("l_dict_capitais_versao_clean.csv")

#  Filtra chunks relevantes
def limpa_texto(texto: str) -> str:
    # Remove caracteres de controle ilegais (exceto \n, \r, \t se desejar)
    texto = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', texto)
    # Remove múltiplos espaços
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def filtra_chunks(documents: list):
    regex_valor_total       = regex_patterns.regex_extrair_valor()
    regex_vagas_total       = regex_patterns.regex_extrair_vagas()
    regex_porcentagem       = regex_patterns.regex_verificar_porcentagem()

    # Filtragem com regex
    chunks_valor_total = [doc for doc in documents if regex_valor_total.search(doc.page_content)]
    chunks_vagas_total = [doc for doc in documents if regex_vagas_total.search(doc.page_content)]
    chunks_porcentagem = [doc for doc in documents if regex_porcentagem.search(doc.page_content)]

    # União sem duplicatas
    chunks_y = list({doc.metadata['id']: doc for doc in (chunks_valor_total + chunks_vagas_total + chunks_porcentagem)}.values())

    # Limpeza dos caracteres indesejados no conteúdo
    for doc in chunks_y:
        doc.page_content = limpa_texto(doc.page_content)

    return chunks_y

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


df_filtrado.to_csv("df_filtrado_capitais_clean.csv")

# Roda o modelo de linguagem
df_filtrado["resultado_llm"] = df_filtrado["texto_completo"].apply(call_gpt_4o_mini)

# Salva o resultados
df_filtrado.to_csv("df_filtrado_full_capitais_clean.csv")
df_filtrado.to_pickle("df_filtrado_full_capitais_clean.pkl")


# Função que remove caracteres não permitidos
def remove_illegal_chars(val):
    if isinstance(val, str):
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', val)
    return val

# Transforma o resultado em DataFrame
df_expanded_llm = df_filtrado['resultado_llm'].apply(pd.Series)


df_cleaned = df_filtrado.drop(columns=['resultado_llm'])

# Em seguida, concatene o DataFrame limpo com as colunas expandidas
df_final = pd.concat([df_cleaned, df_expanded_llm], axis=1)



# Aplica a função em todo o DataFrame
df_limpo = df_final.applymap(remove_illegal_chars)

# Exporta para Excel
df_limpo.to_excel("output_capitai_pt2.xlsx", index=False)
