#%% Imports
from utils import (
    pdf_parser, 
    chunknizer, 
    get_embedding, 
    add_to_chroma, 
    get_chunk_ids,
    call_gpt_4o_mini
)
#%%
import  regex_patterns
import os
import pandas as pd

# %% Contanates
INPUT_PATH_ESTADOS = r"C:\Users\Gabriel\Documents\GitHub\editai_extractor_llm_based\data\input\editais_estados"

# %% Carregamento dos dados
def load_data(input_files: str) -> pd.DataFrame:
    INPUT_PATH = input_files
    l = []
    estados = [nome for nome in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH, nome))]
    for estado in estados:
        pasta = INPUT_PATH + f"\{estado}"
        d = {
            "estado"     : estado, 
            "path_files" : [pasta + f"\{f}" for f in os.listdir(pasta) if f.endswith('.pdf')],
            "files"      : [f for f in os.listdir(pasta) if f.endswith('.pdf')]

        }
        l.append(d)

    return pd.DataFrame(l)

#%%
df_load_data = load_data(input_files=INPUT_PATH_ESTADOS)

#%%
l_documents = []
l_dict      = []
for idx, row in df_load_data.iterrows():
    estado = row["estado"]
    for idx, path in enumerate(row["path_files"]):
        id_doc      = row["files"][idx]
        texto       = pdf_parser(path)
        chunks      = chunknizer(text=texto)
        documents   = get_chunk_ids(chunks=chunks, edital_id=id_doc, uf_edital=estado)
        l_documents.append(documents)
        d = {
            "uf": estado,
            "path_pdf": path,
            "pdf": id_doc,
            "document": documents
        }
        l_dict.append(d)

# %% TESTE
path_teste  = r"C:\Users\Gabriel\Documents\GitHub\editai_extractor_llm_based\data\input\editais_estados\TOCANTINS\TOCANTINS_Edital_Premiação_-_Pontos_e_Pontões_PNCV_1.pdf"
texto_teste = pdf_parser(path_teste)
chunks      = chunknizer(text=texto_teste)
documents   = get_chunk_ids(chunks=chunks, edital_id="TOCANTINS_Edital_Premiação_-_Pontos_e_Pontões_PNCV_1.pdf", uf_edital="TOCANTINS")

# %% Filtra chunks relevantes
def filtra_chunks(documents: list):
    
    regex_valor_total       = regex_patterns.regex_extrair_valor()
    regex_vagas_total       = regex_patterns.regex_extrair_vagas()
    regex_porcentagem       = regex_patterns.regex_verificar_porcentagem()

    # Vamos criar listas de docs filtrados (com metadados)
   
    chunks_valor_total       = [doc for doc in documents if regex_valor_total.search(doc.page_content)]
    chunks_vagas_total       = [doc for doc in documents if regex_vagas_total.search(doc.page_content)]
    chunks_porcentagem       = [doc for doc in documents if regex_porcentagem.search(doc.page_content)]

    # União dos chunks filtrados (evitando duplicatas)
    chunks_y = list({doc.metadata['id']: doc for doc in (chunks_valor_total + chunks_vagas_total + chunks_porcentagem)}.values())

    # Substituir quebras de linha no conteúdo mantendo objetos Document
    for doc in chunks_y:
        doc.page_content = doc.page_content.replace("\n", " ")

    return chunks_y


# %% 
chunks_relevantes = filtra_chunks(documents)
texto_completo = "\n\n".join(chunks_relevantes)
#%%
resultados = call_gpt_4o_mini(texto_completo)
# %%
