
#%%
from utils import pdf_parser, chunknizer, get_embedding, add_to_chroma, get_chunk_ids
import tiktoken
import  regex_patterns
import pandas as pd 

#%%
pdf = "/Users/gabrielribeirobizerril/Documents/GitHub/llm/editai_extractor_llm_based/data/input/capitais/JOÃO PESSOA/2024-07_JOAOPESSOA_SUBSÍDIO.pdf"
text_parsed = pdf_parser(pdf)
#%%
# %% Chunkeia 
chunks      = chunknizer(text=text_parsed)
# documents   = get_chunk_ids(chunks=chunks)

#%% Filtra
regex_acoes_afirmativas     = regex_patterns.regex_verificar_acoes_afirmativas()
regex_valor_total           = regex_patterns.regex_extrair_valor()
regex_vagas_total           = regex_patterns.regex_extrair_vagas()
regex_porcentagem           = regex_patterns.regex_verificar_porcentagem()


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
chunks_full = [i.replace("\n", " ") for i in chunks]

#%%
chunks_full = " ".join(chunks_full)
#%%
def contar_tokens_prompt(chunk_texto: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)

    # Prompt base (como será enviado ao modelo)
    template_prompt = f"""

    A seguir está um fragmento de um edital público. Extraia as informações abaixo exatamente como aparecem no texto. Se a informação não estiver presente, use "NaN".

    Texto:
    {chunk_texto}

    Perguntas:
    - valor_total: Qual o valor total do edital?
    - cotas_negras: Qual o percentual de cotas para pessoas negras?
    - cotas_indigenas: Qual o percentual de cotas para pessoas indígenas?
    - cotas_pcd: Qual o percentual de cotas para pessoas com deficiência (pcd)?
    - vagas_totais: Quantos projetos/propostas/vagas serão contemplados/disponibilizados/selecionados?

    Retorne um JSON com as chaves exatamente assim:
    {{ 
        "valor_total": "...",
        "cotas_negras": "...",
        "cotas_indigenas": "...",
        "cotas_pcd": "...",
        "vagas_totais": "..."
    }}
    """
           

    return len(enc.encode(template_prompt.strip()))
# %%
def estimar_custo_gpt(tokens_input, tokens_output_est=150, tokens_embedding=None, model="gpt-4o-mini"):
    # Garantir que os tokens são números
    tokens_input = int(tokens_input)
    tokens_output_est = int(tokens_output_est)
    tokens_embedding = int(tokens_embedding) if tokens_embedding is not None else tokens_input  # default = texto do chunk

    # Preços por 1.000 tokens
    if model == "gpt-4o-mini":
        preco_input = 0.0005
        preco_output = 0.0015
    elif model == "gpt-4o":
        preco_input = 0.005
        preco_output = 0.015
    else:
        raise ValueError("Modelo não reconhecido")

    preco_embedding = 0.00002  # text-embedding-3-small (OpenAI)

    custo_input     = (tokens_input / 1000) * preco_input
    custo_output    = (tokens_output_est / 1000) * preco_output
    custo_embedding = (tokens_embedding / 1000) * preco_embedding

    custo_total = custo_input + custo_output + custo_embedding
    return round(custo_total, 6)
# # %%
# l = []
# for chunk in chunks_y:
#     token_input = contar_tokens_prompt(chunk)
#     token_embedding = token_input  # ou use outra função se quiser contar separado

#     d = {
#         "chunk"        : chunk,
#         "n_tokens"     : token_input,
#         "custo_total"  : estimar_custo_gpt(token_input, tokens_embedding=token_embedding),
#         "custo_llm"    : estimar_custo_gpt(token_input, tokens_embedding=0),
#         "custo_embed"  : round((token_embedding / 1000) * 0.00002, 6)
#     }
#     l.append(d)
# %%
token_input = contar_tokens_prompt(chunks_full)
token_embedding = token_input 
d = {
        "n_tokens"     : token_input,
        "custo_total"  : estimar_custo_gpt(token_input, tokens_embedding=token_embedding),
        "custo_llm"    : estimar_custo_gpt(token_input, tokens_embedding=0),
        "custo_embed"  : round((token_embedding / 1000) * 0.00002, 6)
    }
# %%
