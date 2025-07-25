
#%%
from utils import pdf_parser, chunknizer, get_embedding, add_to_chroma, get_chunk_ids
import tiktoken
import  regex_patterns
import pandas as pd 

#%%
pdf = "./../data/data_test/TOCANTINS_Edital_TCC_-_Pontos_PNCV_1-1-18.pdf"

text_parsed = pdf_parser(pdf)
#%%
# %% Chunkeia 
chunks      = chunknizer(text=text_parsed)
# documents   = get_chunk_ids(chunks=chunks)

#%% Filtra
regex_acoes_afirmativas = regex_patterns.regex_verificar_acoes_afirmativas()
regex_valor_total       = regex_patterns.regex_extrair_valor()
regex_vagas_total       = regex_patterns.regex_extrair_vagas()


chunks_acoes_afirmativas = [
    chunk for chunk in chunks if regex_acoes_afirmativas.search(chunk)
]

chunks_valor_total = [
    chunk for chunk in chunks if regex_valor_total.search(chunk)
]

chunks_vagas_total = [
    chunk for chunk in chunks if regex_vagas_total.search(chunk)
]

chunks_y = chunks_acoes_afirmativas + chunks_valor_total + chunks_vagas_total
#%%
def contar_tokens(texto, model="text-embedding-3-small"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(texto))
# %%
def estimar_custo_embedding(texto):
    tokens = contar_tokens(texto)
    return tokens / 1000 * 0.00002

# %%
l = []
for chunk in chunks_y:
    d = {
        "chunk": chunk,
        "n_tokens": contar_tokens(chunk),
        "custo": estimar_custo_embedding(chunk)
    }
    l.append(d)
# %%
