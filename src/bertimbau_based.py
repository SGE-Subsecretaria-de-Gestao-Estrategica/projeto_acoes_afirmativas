#%% IMPORTS
from    utils       import pdf_parser, chunknizer, filter_regex
from    transformers   import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import  regex_patterns
import  pandas as pd
import  os, re
import  torch

# %% CARREGA A BASE
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


# %% INTERPRETA O PDF
pdf_parsed = pdf_parser(pdf_path=PDF_PATH)

#%% CHUNKINEZER 
chunks  = chunknizer(text=pdf_parsed, chunk_size=800, chunk_overlap=100)
# %% PRIMEIRO FILTRO DO REGEX
regex_acoes_afirmativas = regex_patterns.regex_verificar_acoes_afirmativas()

chunks_acoes_afirmativas = [
    chunk for chunk in chunks if regex_acoes_afirmativas.search(chunk)
]

#%% Perguntas
questions = {
        "negros": [
            "Qual a porcentagem de vagas para pessoas negras?",
            "Quantos por cento são reservados para negros?",
            "Qual o percentual para pessoas pretas ou pardas?",
            "Qual percentual de vagas disponibilizadas para negros?"
        ],
        "indigenas": [
            "Qual a porcentagem de vagas para indígenas?",
            "Quantos por cento são reservados para povos indígenas?",
            "Qual o percentual para pessoas indígenas?"
        ],
        "pcd": [
            "Qual a porcentagem de vagas para pessoas com deficiência?",
            "Quantos por cento são reservados para PCD?",
            "Qual o percentual para deficientes?"
            ]
}

# %% SETTING DO MODELO

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU (MPS) disponível. Usando MPS.")
elif torch.cuda.is_available(): # Para compatibilidade futura ou em outras máquinas
    device = torch.device("cuda")
    print("GPU (CUDA) disponível. Usando CUDA.")
else:
    device = torch.device("cpu")
    print("Nenhuma GPU disponível. Usando CPU.")

# model     = AutoModelForQuestionAnswering.from_pretrained("neuralmind/bert-large-portuguese-cased").to(device)
# tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")


#%% Perguntas
questions_map = {
    "negros": [
        "Qual a porcentagem de vagas para pessoas negras?",
        "Quantos por cento são reservados para negros?",
        "Qual o percentual para pessoas pretas ou pardas?",
        "Qual percentual de vagas disponibilizadas para negros?"
    ],
    "indigenas": [
        "Qual a porcentagem de vagas para indígenas?",
        "Quantos por cento são reservados para povos indígenas?",
        "Qual o percentual para pessoas indígenas?"
    ],
    "pcd": [
        "Qual a porcentagem de vagas para pessoas com deficiência?",
        "Quantos por cento são reservados para PCD?",
        "Qual o percentual para deficientes?"
    ],
    "vagas": [
        "Quantos projetos no total?",
        "Número total de projetos",
        "Quantos selecionados ao total?",
        "Qual o número total de vagas?",
        "Qual a quantidade total de vagas?"
    ],
    "valor_total": [
        "Qual o valor total do edital?",
        "Qual o valor total de recursos?",
        "Qual o valor total do investimento?",
        "Qual o investimento total previsto?",
        "Qual o orçamento total disponível?"
    ]
}

mapa_palavras_numeros = {
    "um": 1, "dois": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18,
    "dezenove": 19, "vinte": 20, "vinte e um": 21, "vinte e dois": 22, "vinte e tres": 23, "vinte e quatro": 24,
    "vinte e cinco": 25, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80,
    "noventa": 90, "cem": 100
}
chaves_ordenadas = sorted(mapa_palavras_numeros.keys(), key=len, reverse=True)


# %%
def extract_numeric_answer_optimized(context, group, top_k_answers=5, min_score_threshold=0.1):
    """
    Extrai uma resposta numérica para um grupo específico de perguntas de um contexto,
    utilizando o modelo BERTimbau Large para QA e uma lógica de pós-processamento robusta.

    Args:
        context (str): O texto onde a resposta será procurada.
        group (str): O grupo de perguntas (ex: "negros", "vagas", "valor_total").
        top_k_answers (int): Número de melhores respostas para considerar de cada pergunta.
        min_score_threshold (float): Limiar mínimo de score para considerar uma resposta.

    Returns:
        int/float/None: O valor numérico extraído, ou None se nenhum for encontrado.
    """

    if group not in questions_map:
        return None

    candidate_answers = [] # Lista para armazenar as melhores respostas candidatas de todas as perguntas do grupo

    # 1. Processar todas as perguntas do grupo em lote
    group_questions = questions_map[group]
    
    # Preparar inputs para o batch
    # A lista de contextos é repetida para cada pergunta, criando pares (pergunta, contexto)
    inputs_batch = tokenizer(
        group_questions,
        [context for _ in range(len(group_questions))],
        return_tensors = "pt",
        padding    = True,
        truncation = True
    )

    # Mover inputs para a GPU
    inputs_batch = {k: v.to(device) for k, v in inputs_batch.items()}

    # Executar inferência em lote
    with torch.no_grad():
        outputs = model(**inputs_batch)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits


    for i in range(len(group_questions)):
        current_input_ids       = inputs_batch["input_ids"][i]
        current_start_logits    = start_logits[i]
        current_end_logits      = end_logits[i]

        # Encontrar os top-k pares (start, end) com base nos scores combinados
        # Adaptação para pegar os top K ao invés de apenas o melhor
        # (Isso é uma simplificação; para uma implementação robusta, você usaria funções como `top_k` do torch
        # e verificaria se end_index >= start_index e o tamanho do span)
        
        # Obter os top-k índices para início e fim
        top_start_indices = torch.topk(current_start_logits, top_k_answers).indices
        top_end_indices = torch.topk(current_end_logits, top_k_answers).indices

        for start_idx in top_start_indices:
            for end_idx in top_end_indices:
                # Verificar se o span é válido (fim >= início e não muito longo)
                if end_idx >= start_idx and (end_idx - start_idx + 1) < 20: # Limite arbitrário de 20 tokens para o span
                    score = current_start_logits[start_idx].item() + current_end_logits[end_idx].item()
                    
                    if score >= min_score_threshold: # Filtrar por score mínimo
                        answer_span_ids = current_input_ids[start_idx : end_idx + 1]
                        answer_text = tokenizer.decode(answer_span_ids).strip()
                        
                        # Adicionar à lista de candidatos
                        candidate_answers.append({
                            'text': answer_text,
                            'score': score
                        })
        
    candidate_answers = sorted(candidate_answers, key=lambda x: x['score'], reverse=True)

    # 3. Pós-processamento e Extração Numérica a partir dos candidatos
    for answer_candidate in candidate_answers:
        text_answer = answer_candidate['text']
        score = answer_candidate['score'] # Podemos usar o score para priorizar

        # Tentar extrair porcentagem
        if "%" in text_answer:
            match = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*%', text_answer) # Captura float também
            if match:
                try:
                    return float(match.group(1).replace(',', '.'))
                except ValueError:
                    continue # Não conseguiu converter, tenta próximo candidato
        
        # Tentar extrair valor monetário (R$)
        match_valor = re.search(r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)', text_answer, re.IGNORECASE)
        if match_valor:
            valor_str = match_valor.group(1).replace('.', '').replace(',', '.')
            try:
                return float(valor_str)
            except ValueError:
                continue

        # Tentar extrair números gerais (inteiros ou decimais)
        match_num = re.search(r'(\d{1,}(?:[.,]\d+)?)', text_answer) # Captura números com ou sem ponto/vírgula
        if match_num:
            num_str = match_num.group(1).replace(',', '.')
            try:
                return float(num_str)
            except ValueError:
                continue

        # Tentar converter palavras para números (se não encontrou dígito)
        # Percorre as chaves ordenadas (maior para menor, para "vinte e um" antes de "vinte")
        for palavra_numerica in chaves_ordenadas:
            # Verifica se a palavra numérica está presente na string de entrada (case-insensitive)
            if re.search(r'\b' + re.escape(palavra_numerica) + r'\b', text_answer, re.IGNORECASE):
                return mapa_palavras_numeros[palavra_numerica] # Retorna o valor numérico correspondente
    
    return None # Nenhuma resposta numérica válida encontrada

#%% mainrun
t = []
for chunk in chunks_acoes_afirmativas:
    a = {}
    a["chunk"] = chunk
    a["group"] = "negros"
    a["r"]     = extract_numeric_answer_optimized(context=chunk,group="negros")

    t.append(a)
# %% TESTE

texto = '7. COTAS\n7.1 Ficam garantidas, conforme descrito no Anexo 1, cotas neste edital para:\n\na. pessoas negras (pretas e pardas): 25% (vinte e cinco por cento) das\n\nvagas;\n\nb. pessoas indígenas: 10% (dez por cento) das vagas;\nc. pessoas com deficiência: 5% (cinco por cento) das vagas.\n\n7.2 As cotas serão destinadas às entidades que possuam quadro de dirigentes\nmajoritariamente (cinquenta por cento mais um) composto por pessoas negras,\nindígenas ou com deficiência, ou que tenham pessoas negras, indígenas ou com\ndeficiência na maioria (cinquenta por cento mais um) das posições de liderança\n(coordenação/direção) no projeto cultural.'

lista_resultados_para_df = [] # Lista para armazenar dicionários para o DataFrame

for question in questions_map["negros"]:
    inputs = tokenizer(question, texto, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # Boa prática para inferência
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Para a melhor resposta:
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    
    # Calcular o score combinado
    # Mova os logits para a CPU antes de chamar .item() para evitar erros se estiver no MPS
    score = start_logits[0, start_index].item() + end_logits[0, end_index].item()
    
    # Decodificar a resposta
    # answer_ids é o input_ids[0] do batch (que você já tinha)
    # Certifique-se de que answer_ids esteja na CPU para o tokenizer.decode()
    answer_ids_cpu = inputs["input_ids"][0].cpu() 
    resposta_encontrada = tokenizer.decode(answer_ids_cpu[start_index : end_index + 1])

    # Adicionar os dados ao dicionário para o DataFrame
    lista_resultados_para_df.append({
        "Pergunta": question,
        "Resposta Encontrada": resposta_encontrada,
        "Score": f"{score:.4f}" # Formatar para 4 casas decimais para melhor visualização
    })

# Criar o DataFrame
df_resultados = pd.DataFrame(lista_resultados_para_df)


# %%
for item in a:
    print(f"Pergunta: {item['pergunta']}")
    print(f"Start Logits (CPU): {item['current_start_logits'].shape}")
    print(f"End Logits (CPU): {item['current_end_logits'].shape}")
    # Você pode inspecionar os valores aqui se quiser, ex: print(item['current_start_logits'])
    print("-" * 20)

# %%
model_name = "neuralmind/bert-large-portuguese-cased"
qa_pipeline = pipeline(
    "question-answering",
    model=model_name
    # Se você quiser forçar o MPS, pode tentar: device="mps"
    # Mas é melhor deixar o pipeline detectar automaticamente, ou usar o device ID:
    # device=0 if torch.backends.mps.is_available() else -1 # 0 para GPU, -1 para CPU
)

print(f"Pipeline carregado. Usando modelo: {model_name}")
print(f"Dispositivo inferido pelo pipeline: {qa_pipeline.device}") # Isso mostrará se está usando CPU ou GPU/MPS

# --- 2. PERGUNTAS DE TESTE ---
questions_map = {
    "negros": [
        "Qual a porcentagem de vagas para pessoas negras?",
        "Quantos por cento são reservados para negros?",
        "Qual o percentual para pessoas pretas ou pardas?",
        "Qual percentual de vagas disponibilizadas para negros?"
    ],
    # Você pode adicionar outros grupos aqui se quiser testá-los
}

# --- 3. TEXTO DE CONTEXTO ---
texto = '7. COTAS\n7.1 Ficam garantidas, conforme descrito no Anexo 1, cotas neste edital para:\n\na. pessoas negras (pretas e pardas): 25% (vinte e cinco por cento) das\n\nvagas;\n\nb. pessoas indígenas: 10% (dez por cento) das vagas;\nc. pessoas com deficiência: 5% (cinco por cento) das vagas.\n\n7.2 As cotas serão destinadas às entidades que possuam quadro de dirigentes\nmajoritariamente (cinquenta por cento mais um) composto por pessoas negras,\nindígenas ou com deficiência, ou que tenham pessoas negras, indígenas ou com\ndeficiência na maioria (cinquenta por cento mais um) das posições de liderança\n(coordenação/direção) no projeto cultural.'

# --- 4. EXECUTAR TESTES E COLETAR RESULTADOS ---
lista_resultados_para_df = []

print("\nExecutando QA com pipeline...")
for question in questions_map["negros"]:
    # O pipeline faz a tokenização, inferência e decodificação internamente
    result = qa_pipeline(question=question, context=texto)

    # O 'result' do pipeline é um dicionário com 'score', 'start', 'end', 'answer'
    lista_resultados_para_df.append({
        "Pergunta": question,
        "Resposta Encontrada": result['answer'],
        "Score": f"{result['score']:.4f}" # Formatar para 4 casas decimais
    })

# Criar o DataFrame
df_resultados = pd.DataFrame(lista_resultados_para_df)

# %%
# ... (Seu código de configuração do modelo e tokenizer, questions_map, mapa_palavras_numeros) ...

def extract_numeric_answer_optimized_with_priority(context, group, top_k_answers=10, min_score_threshold=0.0):
    if group not in questions_map:
        return None

    candidate_spans_with_scores = []

    group_questions = questions_map[group]
    
    inputs_batch = tokenizer(
        group_questions,
        [context for _ in range(len(group_questions))],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs_batch = {k: v.to(device) for k, v in inputs_batch.items()}

    with torch.no_grad():
        outputs = model(**inputs_batch)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    for i in range(len(group_questions)):
        current_input_ids = inputs_batch["input_ids"][i]
        current_start_logits = start_logits[i]
        current_end_logits = end_logits[i]

        start_probs = torch.softmax(current_start_logits, dim=0)
        end_probs = torch.softmax(current_end_logits, dim=0)

        start_indexes = torch.topk(start_probs, top_k_answers).indices.tolist()
        end_indexes = torch.topk(end_probs, top_k_answers).indices.tolist()
        
        # Considerar todas as combinações válidas de início e fim
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if not (end_idx >= start_idx and (end_idx - start_idx + 1) < 20):
                    continue
                if current_input_ids[start_idx] == tokenizer.cls_token_id or \
                   current_input_ids[end_idx] == tokenizer.sep_token_id or \
                   current_input_ids[start_idx] == tokenizer.pad_token_id:
                    continue

                score = current_start_logits[start_idx].item() + current_end_logits[end_idx].item()
                
                if score >= min_score_threshold:
                    answer_span_ids = current_input_ids[start_idx : end_idx + 1]
                    answer_text = tokenizer.decode(answer_span_ids).strip()
                    
                    candidate_spans_with_scores.append({
                        'text': answer_text,
                        'score': score,
                        'question_idx': i
                    })
    
    candidate_spans_with_scores = sorted(candidate_spans_with_scores, key=lambda x: x['score'], reverse=True)

    # --- NOVA LÓGICA DE EXTRAÇÃO COM PRIORIDADE PARA PERCENTUAIS ---
    for answer_candidate in candidate_spans_with_scores:
        text_answer = answer_candidate['text']

        # 1. Tentar extrair porcentagem como ALTA PRIORIDADE para grupos de cotas
        if group in ["negros", "indigenas", "pcd"]:
            # Prioriza padrões "XX%" ou "XX %" ou "XX por cento"
            match_perc = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*(?:%|por cento)', text_answer, re.IGNORECASE)
            if match_perc:
                try:
                    val = float(match_perc.group(1).replace(',', '.'))
                    if 0 <= val <= 100:
                        return val
                except ValueError:
                    pass
        
        # 2. Se não encontrou porcentagem para grupos de cotas, ou se é outro grupo,
        # tentar valor monetário ou número geral.
        # (Manter a ordem de prioridade geral: R$, número, palavra-número)
        
        # Tentar extrair valor monetário
        match_valor = re.search(r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)', text_answer, re.IGNORECASE)
        if match_valor:
            valor_str = match_valor.group(1).replace('.', '').replace(',', '.')
            try:
                return float(valor_str)
            except ValueError:
                pass

        # Tentar extrair números gerais
        match_num = re.search(r'(\d{1,}(?:[.,]\d+)?)', text_answer)
        if match_num:
            num_str = match_num.group(1).replace(',', '.')
            try:
                val = float(num_str)
                # Se for um grupo de porcentagem e o número não está entre 0-100, descartar.
                if group in ["negros", "indigenas", "pcd"] and not (0 <= val <= 100):
                    pass # Não retornar, tentar próximo candidato
                else:
                    return val
            except ValueError:
                pass

        # Tentar converter palavras para números
        for palavra_numerica in chaves_ordenadas:
            if re.search(r'\b' + re.escape(palavra_numerica) + r'\b', text_answer, re.IGNORECASE):
                val = mapa_palavras_numeros[palavra_numerica]
                # Se for um grupo de porcentagem e o número não está entre 0-100, descartar.
                if group in ["negros", "indigenas", "pcd"] and not (0 <= val <= 100):
                    pass # Não retornar, tentar próximo candidato
                else:
                    return val
    
    return None

# --- Exemplo de uso para testar ---
texto_cotas = '7. COTAS\n7.1 Ficam garantidas, conforme descrito no Anexo 1, cotas neste edital para:\n\na. pessoas negras (pretas e pardas): 25% (vinte e cinco por cento) das\n\nvagas;\n\nb. pessoas indígenas: 10% (dez por cento) das vagas;\nc. pessoas com deficiência: 5% (cinco por cento) das vagas.\n\n7.2 As cotas serão destinadas às entidades que possuam quadro de dirigentes\nmajoritariamente (cinquenta por cento mais um) composto por pessoas negras,\nindígenas ou com deficiência, ou que tenham pessoas negras, indígenas ou com\ndeficiência na maioria (cinquenta por cento mais um) das posições de liderança\n(coordenação/direção) no projeto cultural.'

print("\n--- Teste com Prioridade para Percentuais (Abordagem Manual) ---")
perc_negros = extract_numeric_answer_optimized_with_priority(texto_cotas, "negros")
print(f"Porcentagem para negros: {perc_negros}") # Deve ser 25.0

perc_indigenas = extract_numeric_answer_optimized_with_priority(texto_cotas, "indigenas")
print(f"Porcentagem para indígenas: {perc_indigenas}") # Deve ser 10.0

perc_pcd = extract_numeric_answer_optimized_with_priority(texto_cotas, "pcd")
print(f"Porcentagem para PCD: {perc_pcd}") # Deve ser 5.0

num_vagas = extract_numeric_answer_optimized_with_priority(texto_cotas, "vagas")
print(f"Número de vagas: {num_vagas}") # Deve ser 15.0
# %%


import re
import torch
import pandas as pd # Importe pandas se for usar para a tabela de resultados
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# --- CONFIGURAÇÃO INICIAL (Repita do seu script principal, garantindo que seja executado apenas uma vez) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU (MPS) disponível. Usando MPS.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU (CUDA) disponível. Usando CUDA.")
else:
    device = torch.device("cpu")
    print("Nenhuma GPU disponível. Usando CPU.")

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-large-portuguese-cased")
model = AutoModelForQuestionAnswering.from_pretrained("neuralmind/bert-large-portuguese-cased").to(device)
print(f"Modelo movido para o dispositivo: {device}")
print("Modelos carregados com sucesso!")

# --- Mapeamentos de Perguntas e Palavras para Números (Mantenha como estão) ---
questions_map = {
    "negros": [
        "Qual a porcentagem de vagas para pessoas negras?",
        "Quantos por cento são reservados para negros?",
        "Qual o percentual para pessoas pretas ou pardas?",
        "Qual percentual de vagas disponibilizadas para negros?"
    ],
    "indigenas": [
        "Qual a porcentagem de vagas para indígenas?",
        "Quantos por cento são reservados para povos indígenas?",
        "Qual o percentual para pessoas indígenas?"
    ],
    "pcd": [
        "Qual a porcentagem de vagas para pessoas com deficiência?",
        "Quantos por cento são reservados para PCD?",
        "Qual o percentual para deficientes?"
    ],
    "vagas": [
        "Quantos projetos no total?",
        "Número total de projetos",
        "Quantos selecionados ao total?",
        "Qual o número total de vagas?",
        "Qual a quantidade total de vagas?"
    ],
    "valor_total": [
        "Qual o valor total do edital?",
        "Qual o valor total de recursos?",
        "Qual o valor total do investimento?",
        "Qual o investimento total previsto?",
        "Qual o orçamento total disponível?"
    ]
}

mapa_palavras_numeros = {
    "um": 1, "dois": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18,
    "dezenove": 19, "vinte": 20, "vinte e um": 21, "vinte e dois": 22, "vinte e tres": 23, "vinte e quatro": 24,
    "vinte e cinco": 25, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80,
    "noventa": 90, "cem": 100
}
chaves_ordenadas = sorted(mapa_palavras_numeros.keys(), key=len, reverse=True)


# --- FUNÇÃO PRINCIPAL APRIMORADA ---
def extract_numeric_answer_optimized_with_priority(
    context,
    group,
    num_top_spans_to_consider=30, # Aumentado para buscar mais candidatos
    max_answer_length_tokens=20, # Comprimento máximo aceitável para a resposta em tokens
    min_logit_score_threshold=-10.0 # Ajustado para ser mais permissivo na seleção inicial de spans
):
    if group not in questions_map:
        return None

    candidate_spans_with_scores = []

    group_questions = questions_map[group]
    
    inputs_batch = tokenizer(
        group_questions,
        [context for _ in range(len(group_questions))],
        return_tensors="pt",
        padding=True,
        truncation=True, # Mantém truncamento, caso seu chunk seja muito grande
        max_length=512
    )
    inputs_batch = {k: v.to(device) for k, v in inputs_batch.items()}

    with torch.no_grad():
        outputs = model(**inputs_batch)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    for i in range(len(group_questions)):
        current_input_ids = inputs_batch["input_ids"][i]
        current_start_logits = start_logits[i]
        current_end_logits = end_logits[i]

        # Pegar os top N índices com base nos logits crus (não nas probabilidades)
        # para ter um leque maior de candidatos
        start_indexes = torch.topk(current_start_logits, num_top_spans_to_consider).indices.tolist()
        end_indexes = torch.topk(current_end_logits, num_top_spans_to_consider).indices.tolist()
        
        # Combinar índices para encontrar spans potenciais
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                # 1. Validação básica do span
                if not (end_idx >= start_idx and (end_idx - start_idx + 1) <= max_answer_length_tokens):
                    continue # Span inválido ou muito longo
                
                # 2. Filtrar spans que contêm apenas tokens especiais no início/fim
                # IDs de tokens especiais para BERTimbau: CLS=101, SEP=102, PAD=0
                if current_input_ids[start_idx] in [tokenizer.cls_token_id, tokenizer.pad_token_id] or \
                   current_input_ids[end_idx] in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    continue

                # Calcular score combinado (soma dos logits crus)
                score = current_start_logits[start_idx].item() + current_end_logits[end_idx].item()
                
                if score >= min_logit_score_threshold: # Filtro de score mais permissivo aqui
                    # Decodificar o span. skip_special_tokens=True é crucial aqui!
                    answer_span_ids = current_input_ids[start_idx : end_idx + 1]
                    answer_text = tokenizer.decode(answer_span_ids, skip_special_tokens=True).strip()
                    
                    # 3. Filtrar respostas vazias ou que consistem apenas em pontuação/ruído
                    # Isso é importante para descartar decodificações estranhas
                    if len(answer_text) > 0 and not re.fullmatch(r'[\s.,\-\/#!$%\^&*;:{}=\-_`~()\[\]]+', answer_text):
                        candidate_spans_with_scores.append({
                            'text': answer_text,
                            'score': score,
                            'question_idx': i # Mantém o índice da pergunta, se necessário para depuração
                        })
    
    # Ordenar todos os spans válidos encontrados pelo score (do maior para o menor)
    candidate_spans_with_scores = sorted(candidate_spans_with_scores, key=lambda x: x['score'], reverse=True)

    # --- Lógica de Pós-processamento e Extração Numérica com Prioridade ---
    for answer_candidate in candidate_spans_with_scores:
        text_answer = answer_candidate['text']
        
        # 1. PRIORIDADE MÁXIMA para Percentuais para os grupos de cotas
        if group in ["negros", "indigenas", "pcd"]:
            # Procura por número seguido de % ou "por cento"
            match_perc = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*(?:%|por\s*cento)', text_answer, re.IGNORECASE)
            if match_perc:
                try:
                    val = float(match_perc.group(1).replace(',', '.'))
                    # Validação adicional para garantir que é uma porcentagem razoável
                    if 0 <= val <= 100:
                        return val
                except ValueError:
                    pass # Tentar o próximo padrão
        
        # 2. Prioridade para valores monetários se for o grupo "valor_total"
        if group == "valor_total" or "R$" in text_answer:
            match_valor = re.search(r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)', text_answer, re.IGNORECASE)
            if match_valor:
                valor_str = match_valor.group(1).replace('.', '').replace(',', '.')
                try:
                    return float(valor_str)
                except ValueError:
                    pass

        # 3. Prioridade para números gerais para o grupo "vagas" ou se for apenas um número
        # Esta é a parte que deve capturar o "15" para vagas.
        if group == "vagas" or re.search(r'\d', text_answer): # Se tem dígito no texto
            match_num = re.search(r'(\d{1,}(?:[.,]\d+)?)', text_answer)
            if match_num:
                num_str = match_num.group(1).replace(',', '.')
                try:
                    val = float(num_str)
                    # Se o grupo é de porcentagem, mas não achamos o "%" e o número não é 0-100, descartar.
                    if group in ["negros", "indigenas", "pcd"] and not (0 <= val <= 100):
                        pass
                    # Se for "vagas", retorne o número diretamente.
                    elif group == "vagas":
                        return val
                    # Caso contrário, se for um número geral e não um grupo de porcentagem problemático.
                    else:
                        return val
                except ValueError:
                    pass

        # 4. Tentar converter palavras para números (última opção)
        # Verificar se a palavra numérica está no texto decodificado
        for palavra_numerica in chaves_ordenadas:
            # Use \b para garantir que seja a palavra inteira
            if re.search(r'\b' + re.escape(palavra_numerica) + r'\b', text_answer.lower()):
                val = mapa_palavras_numeros[palavra_numerica]
                # Validação para evitar números "palavra" grandes para cotas
                if group in ["negros", "indigenas", "pcd"] and not (0 <= val <= 100):
                    pass
                else:
                    return val
    
    return None # Retorna None se nenhuma resposta numérica válida for encontrada

# --- EXEMPLO DE USO PARA TESTAR ---

# %%
