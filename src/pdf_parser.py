#%% IMPORTS
from pdfminer.high_level import extract_text
from ctransformers import AutoModelForCausalLM
import os

#%%
def extract_text_from_pdf_miner(pdf_path: str):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Erro ao extrair texto do PDF com pdfminer.six: {e}")
        return None

# %%
def load_local_llm(model_path, model_type="mistral", gpu_layers=-1):
    """
    Carrega um modelo LLM localmente usando ctransformers.
    Ajustado para usar a GPU (Apple Silicon).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}. Por favor, baixe o arquivo .gguf.")

    print(f"Carregando modelo LLM de: {model_path}...")
    print(f"Tentando carregar {gpu_layers if gpu_layers == -1 else str(gpu_layers)} camadas na GPU...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type=model_type,
        gpu_layers=gpu_layers,
        temperature=0.1,
        max_new_tokens=500
    )
    print("Modelo LLM carregado com sucesso!")
    return llm
# %%
def extract_info_with_llm(llm_model, document_text, info_to_extract):
    """
    Usa a LLM para extrair informações específicas do texto do documento.
    """
    info_list_str = "\n".join([f"- {item}" for item in info_to_extract])

    prompt_template = f"""
    A seguir está um documento de texto. Por favor, extraia as seguintes informações dele.
    Se uma informação não for encontrada no documento, responda com "N/A" para aquele item.
    Responda em formato de lista clara.

    ---
    DOCUMENTO:
    {document_text}
    ---

    INFORMAÇÕES A EXTRAIR:
    {info_list_str}

    RESPOSTA (apenas as informações extraídas):
    """

    print("\nEnviando prompt para a LLM...")
    response = llm_model(prompt_template)
    return response.strip()

def save_text_to_file(text, output_filepath):
    """
    Salva o texto fornecido em um arquivo .txt.
    Cria os diretórios necessários se não existirem.
    """
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório criado: {output_dir}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Texto salvo com sucesso em: {output_filepath}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo {output_filepath}: {e}")

#%% GLOBALS
INFO_WANTED = [
    "Nome do ente",
    "Valor total do edital",
    "Percentual de cotas para pessoas negras",
    "Percentual de cotas para pessoas indígenas",
    "Percentual de cotas para pessoas com deficiência (pcd)",
    "Projetos selecionados"
]

LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

#%% SETTINGS
script_dir    = os.path.dirname(__file__)
edital_target = 'TOCANTINS_Edital_TCC_-_Pontos_PNCV_1.pdf'
PDF_PATH      = os.path.join(
    script_dir,
    '..', 'data', 'data_test', edital_target
)
TEXT_OUTPUT_DIR      = os.path.join(
    script_dir, '..', 'data', 'data_test_output'
)
TEXT_OUTPUT_FILENAME = os.path.basename(PDF_PATH).replace('.pdf', '.txt') # Usa o nome do PDF com .txt

FULL_TEXT_OUTPUT_PATH = os.path.join(TEXT_OUTPUT_DIR, TEXT_OUTPUT_FILENAME)
# %% RUN
# Normaliza o caminho do pdf
PDF_PATH_ABSOLUTE = os.path.abspath(PDF_PATH)

if not os.path.exists(PDF_PATH_ABSOLUTE): # AQUI USAR TABELAS DE CONTROLE DAS FALHAS
    print(f"Erro: O arquivo PDF '{PDF_PATH_ABSOLUTE}' não foi encontrado.")
    print("Por favor, verifique o caminho e a estrutura das pastas.")
else:
    # 1. Etapa
    pdf_text = extract_text_from_pdf_miner(PDF_PATH_ABSOLUTE)

    
    if pdf_text:
        # 2. Salva texto - Isso aqui será opcional
        save_text_to_file(pdf_text, FULL_TEXT_OUTPUT_PATH)
    
        # 3. Carregar o Modelo LLM
        try:
            llm = load_local_llm(LLM_MODEL_PATH)
        except Exception as e:
            print(f"Ocorreu um erro durante o processamento da LLM: {e}")


# %%
