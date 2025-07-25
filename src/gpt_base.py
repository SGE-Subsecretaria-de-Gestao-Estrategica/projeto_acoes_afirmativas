
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
regex_porcentagem       = regex_patterns.regex_verificar_porcentagem()


chunks_acoes_afirmativas = [
    chunk for chunk in chunks if regex_porcentagem.search(chunk)
]

chunks_valor_total = [
    chunk for chunk in chunks if regex_valor_total.search(chunk)
]

chunks_vagas_total = [
    chunk for chunk in chunks if regex_vagas_total.search(chunk)
]

chunks_y = chunks_acoes_afirmativas + chunks_valor_total + chunks_vagas_total

#%% clean
chunks_y = [i.replace("\n", " ") for i in chunks_y]
#%%
def contar_tokens_prompt(chunk_texto: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)

    # Prompt base (como será enviado ao modelo)
    template_prompt = f"""
            Instruções:
            - Os valores extraídos devem estar exatamente como no texto
            - Se a informação não estiver no texto, use "NÃO ENCONTRADO"

            Texto para análise:
            {chunk_texto}

            Responda as pergunta:
            - Qual Valor total do edital?
            - Qual Percentual de cotas para pessoas negras?
            - Qual Percentual de cotas para pessoas indígenas?
            - Qual Percentual de cotas para pessoas com deficiência (pcd)?
            - Quantos projetos/vagas serão disponibilizados no edital?
            """

    return len(enc.encode(template_prompt.strip()))
# %%
def estimar_custo_gpt(tokens_input, tokens_output_est=150, model="gpt-4o-mini"):
    # Garantir que tokens_input é número
    tokens_input = int(tokens_input)
    
    if model == "gpt-4o-mini":
        preco_input = 0.0005  # por 1.000 tokens
        preco_output = 0.0015
    elif model == "gpt-4o":
        preco_input = 0.005
        preco_output = 0.015
    else:
        raise ValueError("Modelo não reconhecido")

    custo = (tokens_input / 1000) * preco_input + (tokens_output_est / 1000) * preco_output
    return round(custo, 6)
# %%
l = []
for chunk in chunks_y:
    token_input = contar_tokens_prompt(chunk)
    d = {
        "chunk"    : chunk,
        "n_tokens" : token_input,
        "custo"    : estimar_custo_gpt(token_input)
    }
    l.append(d)
# %%
