#%% IMPORTS
from pdfminer.high_level import extract_text
# from ctransformers import AutoModelForCausalLM
import os, re, random
from llama_cpp import Llama

#%% REGEX PATTERNS - passar para um código separado depois
#=============================================
regex_acoes_afirmativas = re.compile(r"(afirmativas?|cotas?|ações\s*afirmativas?|políticas\s*afirmativas?|negros?|negras?|pessoas negras|população negra|agentes\s*culturais\s*negros?|indígenas?|quilombolas?|povos\s*tradicionais|comunidades\s*tradicionais|lgbtqia\+?|lgbt\+?|lgbt|lésbicas?|gays?|agentes culturais indígenas?|bissexuais?|transgêneros?|transexuais?|travestis?|intersexos?|não[- ]bináries?|pessoas trans|agentes culturais com deficiência|pessoas?\s*com\s*deficiência|pcd|mulheres?|equidade de gênero|diversidade|inclusão|acessibilidade|cadeirantes?|mobilidade reduzida)", re.IGNORECASE)
regex_valor             = re.compile(r"R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?")
regex_vagas             = re.compile(
    r"(?:\b(?:seleção|total de|pelo menos|reconhecer|selecionar|premiar|serão selecionados|escolhidos|contemplados|destina-se a)\b\s*)?" # Opcional: frases introdutórias
    r"(\d+)\s*(?:\((?:[a-zá-ú\s]+)\))?" # Captura o número em algarismo e opcionalmente o número por extenso entre parênteses
    r"\s+" # Um ou mais espaços
    r"\b(vagas|projetos|propostas|contemplados|selecionados|escolhidos|grupos)\b", # As palavras-chave
    re.IGNORECASE | re.UNICODE # Ignora maiúsculas/minúsculas e lida com caracteres acentuados
)
regex_porcentagem = re.compile(r"\d+\s*%")


# --- CORREÇÃO APLICADA AQUI ---
regex_negros = re.compile(
        r"(negros?|negras?|pessoas?\s*negras|população\s*negra|agentes?\s*culturais\s*negros?|"
        r"agentes\s*culturais\s*de\s*matriz\s*africana|culturas?\s*negras?|matriz\s*africana)",
        re.IGNORECASE
    )

regex_indigenas =  re.compile( 
      r"(indígenas?|povos?\s*indígenas?|agentes?\s*culturais\s*indígenas?|culturas?\s*indígenas?|agentes\s*culturais\s*indígenas?|comunidades\s*indígenas|tradições\s*indígenas|pessoas?\s*indígenas)",
      re.IGNORECASE
    ) 

regex_pcd =  re.compile( 
        r"(pessoas?\s*com\s*deficiência|pcd|agentes?\s*culturais\s*com\s*deficiência|"
        r"agentes culturais pcd|artistas?\s*com\s*deficiência)",
        re.IGNORECASE
    ) 
#=============================================
#%% PARSER
def extract_text_from_pdf_miner(pdf_path: str):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Erro ao extrair texto do PDF com pdfminer.six: {e}")
        return None

#%% CHUNKING
def split_text_into_chunks(text: str, chunk_size: int=2000, chunk_overlap: int=200) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - chunk_overlap)
    return chunks

#%% FILTER CHUNKS
def filter_chunks_by_regex(chunks, regex_list):
    """
    Filtra uma lista de chunks, retornando apenas aqueles que contêm
    pelo menos uma correspondência para qualquer regex na lista fornecida.

    Args:
        chunks (list): Uma lista de strings (os chunks de texto).
        regex_list (list): Uma lista de objetos regex compilados (re.Pattern).

    Returns:
        list: Uma nova lista contendo apenas os chunks filtrados.
    """
    filtered_chunks = []
    print("\nIniciando filtragem de chunks por regex...")
    for i, chunk in enumerate(chunks):
        is_relevant = False
        for regex in regex_list:
            if regex.search(chunk):
                is_relevant = True
                # print(f"  Chunk {i+1} relevante por regex: {regex.pattern[:30]}...") # Para depuração
                break 
        if is_relevant:
            filtered_chunks.append(chunk)
    print(f"Filtragem concluída. De {len(chunks)} chunks, {len(filtered_chunks)} foram considerados relevantes.") # AQUI COLOCAR CONTAGEM PARA FINS DE OTIMIZAÇÃO
    return filtered_chunks
# %%
# def load_local_llm(model_path: str, model_type: str ="mistral", gpu_layers: int =-1, context_length: int = "8192"):
#     """
#     Carrega um modelo LLM localmente usando ctransformers.
#     Ajustado para usar a GPU (Apple Silicon).
#     """
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Modelo não encontrado em: {model_path}. Por favor, baixe o arquivo .gguf.")

#     print(f"Carregando modelo LLM de: {model_path}...")
#     print(f"Tentando carregar {gpu_layers if gpu_layers == -1 else str(gpu_layers)} camadas na GPU...")
#     try:
#         llm = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             model_type=model_type,
#             gpu_layers=gpu_layers,
#             temperature=0.1,
#             max_new_tokens=500,
#             context_length=context_length
#         )
#         print("Modelo LLM carregado com sucesso!")
#     except Exception as e:
#         print(e)
#     return llm

def load_local_llm(model_path: str, n_gpu_layers: int =-1, n_ctx: int = 4096, temperature: float = 0.1, max_tokens: int = 500):
    """
    Carrega um modelo LLM localmente usando llama-cpp-python.
    Ajustado para usar a GPU (Apple Silicon).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}. Por favor, baixe o arquivo .gguf.")

    # Parametros para Llama no llama-cpp-python
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers, # Número de camadas para descarregar na GPU (-1 para todas)
        n_ctx=n_ctx,               # Janela de contexto
        temperature=temperature,
        max_tokens=max_tokens,     # Máximo de tokens a serem gerados
        verbose=True               # Para ver logs detalhados
    )
    print("Modelo LLM carregado com sucesso!")
    return llm


# %% EXTRACTION
def extract_info_with_llm(llm_model, document_text, info_to_extract):
    """
    Usa a LLM para extrair informações específicas do texto do documento.
    """
    info_list_str = "\n".join([f"- {item}" for item in info_to_extract])

    prompt_template = f"""
    A seguir está um documento de texto. Por favor, extraia as seguintes informações dele.
    Se uma informação não for encontrada no documento, responda com "N/A" para aquele item.
    Responda em formato de lista clara (estilo Chave:Valor).

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

# def extract_info_from_chunks(llm_model, chunks: list, info_to_extract: list, temperature: float = 0.1, max_tokens: int = 500):
    
#     # Dicionário de output
#     all_extracted_data_consolidated = {}

#     for item in info_to_extract:
#     # Usamos o próprio item da INFO_WANTED como chave inicial
#         all_extracted_data_consolidated[item] = "N/A" # Assume N/A por padrão

#     for i, chunk in enumerate(chunks):
#         print(f"\nProcessando Chunk {i+1}/{len(chunks)}...")

#         # Passa a lista de perguntas para uma string
#         info_list_str = "\n".join([f"- {item}" for item in info_to_extract])

#         # PROMPT para realizar pergunta 
#         prompt_template = f"""
#         A seguir está um documento de texto. Por favor, extraia as seguintes informações dele.
#         Se uma informação não for encontrada no documento, responda com "N/A" para aquele item.
#         Responda em formato de lista clara (estilo Chave:Valor).

#         ---
#         FRAGMENTO DO DOCUMENTO:
#         {chunk}
#         ---

#         INFORMAÇÕES A EXTRAIR:
#         {info_list_str}

#         RESPOSTA (apenas as informações extraídas para ESTE fragmento):
#         """

#         response_dict = llm_model.create_completion(
#             prompt=prompt_template,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             stop=["---", "FRAGMENTO"] # Adicionado "FRAGMENTO" para parar se o modelo "repetir" a estrutura do prompt
#         )
#         response_text = response_dict["choices"][0]["text"]
        

#         lines = response_text.strip().split('\n')
#         for line in lines:
#             if ':' in line:
#                 key, value = line.split(':', 1)
#                 key = key.strip()
#                 value = value.strip()

#                 # --- Lógica de Consolidação ---
#                 # Se o valor não for "N/A" e a chave ainda não tiver um valor válido ou for "N/A"
#                 if value != "N/A" and (key not in all_extracted_data_consolidated or all_extracted_data_consolidated[key] == "N/A"):
#                     all_extracted_data_consolidated[key] = value
        
#     return all_extracted_data_consolidated

def extract_info_from_chunks(llm_model, chunks: list, info_to_extract: list, temperature: float = 0.1, max_tokens: int = 500):

    # Dicionário para armazenar TODOS os valores válidos encontrados para cada chave
    # Cada chave terá uma LISTA de valores, permitindo múltiplos resultados por item.
    all_extracted_data_list_per_key = {item: [] for item in info_to_extract}
    # Também vamos querer saber de qual chunk veio a informação
    results_by_chunk = [] # Uma lista de dicionários, um para cada chunk processado

    for i, chunk in enumerate(chunks):
        print(f"\nProcessando Chunk {i+1}/{len(chunks)}...")

        # Dicionário temporário para armazenar o que foi extraído NESTE chunk
        current_chunk_data = {"Chunk_ID": i + 1} # Adiciona um ID para o chunk na tabela

        info_list_str = "\n".join([f"- {item}" for item in info_to_extract])
        prompt_template = f"""
        A seguir está um fragmento de um documento. Por favor, extraia as seguintes informações dele.
        Se uma informação não for encontrada NESTE FRAGMENTO, responda com "N/A" para aquele item.
        Responda em formato de lista clara (estilo Chave:Valor).

        ---
        FRAGMENTO DO DOCUMENTO:
        {chunk}
        ---

        INFORMAÇÕES A EXTRAIR:
        {info_list_str}

        RESPOSTA (apenas as informações extraídas para ESTE fragmento):
        """

        response_dict = llm_model.create_completion(
            prompt=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["---", "FRAGMENTO"]
        )
        response_text = response_dict["choices"][0]["text"]

        lines = response_text.strip().split('\n')
        
        # Inicializa as chaves do chunk atual com N/A
        for item in info_to_extract:
            current_chunk_data[item] = "N/A"

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Apenas se a chave for uma das que queremos
                if key in all_extracted_data_list_per_key:
                    if value != "N/A":
                        # Adiciona o valor à lista de valores para essa chave
                        all_extracted_data_list_per_key[key].append(value)
                        # Atualiza o dado específico para este chunk
                        current_chunk_data[key] = value
                    else:
                        # Mantém "N/A" para este chunk se o valor extraído for "N/A"
                        current_chunk_data[key] = "N/A" # Garante que a chave existe mesmo que seja N/A para o chunk

        results_by_chunk.append(current_chunk_data)

    return results_by_chunk # Retorna uma lista de dicionários, um por chunk


def format_as_markdown_table(data: dict) -> str:
    """
    Formata um dicionário de informações extraídas em uma tabela Markdown.
    Args:
        data (dict): Dicionário onde chaves são os nomes das informações
                     e os valores são os dados extraídos.
    Returns:
        str: Uma string formatada como tabela Markdown.
    """
    if not data:
        return "Nenhuma informação extraída para exibir em tabela."

    headers = ["Informação", "Valor Extraído"]
    # Garante que a ordem das chaves na tabela siga a ordem de INFO_WANTED, se possível
    # Ou apenas itera sobre as chaves do dicionário se preferir.
    # Usaremos INFO_WANTED para garantir a ordem e incluir itens não encontrados.
    ordered_keys = INFO_WANTED # Supondo que INFO_WANTED define a ordem desejada

    # Cabeçalho da tabela
    table_str = "| " + " | ".join(headers) + " |\n"
    table_str += "|---" * len(headers) + "|\n"

    # Linhas da tabela
    for key_full_question in ordered_keys:
        # A chave no dicionário pode não ser exatamente a pergunta, se você a "limpar"
        # Garante que a chave no dicionário corresponde à pergunta na INFO_WANTED, mesmo que haja pequenas variações.
        # Uma abordagem mais robusta seria ter um mapeamento explícito entre pergunta e chave interna.
        # Por enquanto, assumimos que as chaves do dicionário são as próprias perguntas de INFO_WANTED.
        value = data.get(key_full_question, "N/A") # Usa .get para lidar com chaves não encontradas

        # Se o valor for uma lista (ex: para ações afirmativas se quisesse múltiplas), transforme em string
        if isinstance(value, list):
            value = ", ".join(value) if value else "N/A"

        table_str += f"| {key_full_question} | {value} |\n"

    return table_str







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
    "Qual Valor total do edital",
    "Qual Percentual de cotas para pessoas negras",
    "Qual Percentual de cotas para pessoas indígenas",
    "Qual Percentual de cotas para pessoas com deficiência (pcd)",
    "Quantos projetos selecionados"
]



#%% SETTINGS
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

# model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_name = "phi-2.Q4_K_M.gguf"

LLM_MODEL_PATH = os.path.join(
    script_dir, '..', 'models', model_name
)
model_type = "phi"
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

        # 3. Dividir o texto em chunks
        chunks = split_text_into_chunks(pdf_text, chunk_size=2000, chunk_overlap=200)
        print(f"\nDocumento dividido em {len(chunks)} chunks.")

        # 4. Filtra os chunks
        all_regex_for_filtering = [
                regex_acoes_afirmativas,
                regex_valor,
                regex_vagas,
                regex_porcentagem,
                regex_negros,
                regex_indigenas,
                regex_pcd
            ]

        filtered_chunks       = filter_chunks_by_regex(chunks, all_regex_for_filtering)

        try:
            llm = load_local_llm(LLM_MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, temperature=0.5, max_tokens=500)
            # 5. Extrair informações DOS CHUNKS FILTRADOS com a LLM
            extracted_data = extract_info_from_chunks(llm, filtered_chunks, INFO_WANTED, temperature=0.1, max_tokens=500)

            print("\n--- Informações Consolidadas Extraídas pela LLM ---")
            final_table = format_as_markdown_table(extracted_data)



        #     print(extracted_data)

        except Exception as e:
            print(e)

    
        # # 3. Carregar o Modelo LLM
        # try:
        #     llm = load_local_llm(LLM_MODEL_PATH)

        #     # 4. Extrair informações com a LLM
        #     extracted_data = extract_info_with_llm(llm, pdf_text, INFO_WANTED)

        #     print("\n--- Informações Extraídas pela LLM ---")

        except Exception as e:
            print(f"Ocorreu um erro durante o processamento da LLM: {e}")


# %%
