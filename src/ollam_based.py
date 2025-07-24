#%%
import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from utils import pdf_parser, chunknizer
# %%

llm = OllamaLLM(
    model='deepseek-r1:1.5b',
    temperature=0
)

#%%
script_dir    = os.path.dirname(__file__)
edital_target = 'TOCANTINS_Edital_TCC_-_Pontos_PNCV_1-1-18.pdf'
PDF_PATH      = os.path.join(
    script_dir,
    '..', 'data', 'data_test', edital_target
)
TEXT_OUTPUT_DIR      = os.path.join(
    script_dir, '..', 'data', 'data_test_output'
)
TEXT_OUTPUT_FILENAME = os.path.basename(PDF_PATH).replace('.pdf', '.txt') # Usa o nome do PDF com .txt

FULL_TEXT_OUTPUT_PATH = os.path.join(TEXT_OUTPUT_DIR, TEXT_OUTPUT_FILENAME)

# %%
pdf_parsed = pdf_parser(pdf_path=PDF_PATH)
pdf_chunks = chunknizer(text=pdf_parsed, chunk_size=800, chunk_overlap=100)


#%% Perguntas
INFO_WANTED = [
    "Qual Valor total do edital?",
    "Qual Percentual de cotas para pessoas negras?",
    "Qual Percentual de cotas para pessoas indígenas?",
    "Qual Percentual de cotas para pessoas com deficiência (pcd)?",
    "Quantos projetos serão selecionados?"
]
# %%

info_list_str = "\n".join([f"- {item}" for item in INFO_WANTED])
# %%
text = """
            7. COTAS
            O edital irá disponibilizar 2400 projetos
            7.1 Ficam garantidas, conforme descrito no Anexo 1, cotas neste edital para:
            a. pessoas negras(pretas e pardas): 25% (vinte e cincoporcento)das vagas;
            b. pessoas indígenas: 10% (dez por cento) das vagas;
            c. pessoas com deficiência: 5% (cinco por cento) das vagas.
            7.2 As cotas serão destinadas às entidades que possuam quadro de dirigentes majoritariamente (cinquenta por cento mais um) composto por pessoas negras, indígenas ou com deficiência, ou que tenham pessoas negras, indígenas ou com deficiência na maioria (cinquenta por cento mais um) das posições de liderança (coordenação/direção) no projeto cultural.
            7.3 As pessoas físicas que compõem a direção da entidade proponente ou da equipe do projeto devem se submeter aos regramentos descritos neste Edital, inclusive quanto ao procedimento de heteroidentificação.

"""

template_prompt   = """
ANÁLISE DE EDITAL - EXTRATIVO ESTRITO

Instruções:
1. Responda APENAS com o JSON solicitado
2. Valores devem ser EXATAMENTE como no texto
3. Se não encontrar, use "NÃO ENCONTRADO"

Texto para análise:
{text}

Extraia:
- Percentual de cotas para pessoas negras (busque por "negras" e "%")
- Percentual de cotas para indígenas (busque por "indígenas" e "%")
- Percentual para PCDs (busque por "deficiência" e "%")
- Número total de projetos (busque por "projetos" seguido de número)
"""
#%%
prompt = PromptTemplate(
    input_variables=['text'],
    template=template_prompt
)

chain = prompt | llm | JsonOutputParser()

# %%
response = chain.invoke({'text': text})

# %%
