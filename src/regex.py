import re

def regex_verificar_acoes_afirmativas():
    """
    Verifica termos relacionados a ações afirmativas e diversidade.
    Retorna True se uma correspondência for encontrada, False caso contrário.
    """
    regex_acoes_afirmativas = re.compile(r"(afirmativas?|cotas?|ações\s*afirmativas?|políticas\s*afirmativas?|negros?|negras?|pessoas negras|população negra|agentes\s*culturais\s*negros?|indígenas?|quilombolas?|povos\s*tradicionais|comunidades\s*tradicionais|lgbtqia\+?|lgbt\+?|lgbt|lésbicas?|gays?|agentes culturais indígenas?|bissexuais?|transgêneros?|transexuais?|travestis?|intersexos?|não[- ]bináries?|pessoas trans|agentes culturais com deficiência|pessoas?\s*com\s*deficiência|pcd|mulheres?|equidade de gênero|diversidade|inclusão|acessibilidade|cadeirantes?|mobilidade reduzida)", re.IGNORECASE)
    return regex_acoes_afirmativas

def regex_extrair_valor():
    """
    Extrai valores monetários (ex: R$ 1.234,56).
    Retorna uma lista de todos os valores encontrados, ou uma lista vazia se nenhum for encontrado.
    """
    regex_valor = re.compile(r"R\$\s*\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?")
    return regex_valor

def regex_extrair_vagas():
    """
    Extrai números de 'vagas', 'projetos', 'propostas', etc.
    Retorna uma lista de tuplas, onde cada tupla contém (número, palavra-chave),
    ou uma lista vazia se nenhuma correspondência for encontrada.
    """
    regex_vagas = re.compile(
        r"(?:\b(?:seleção|total de|pelo menos|reconhecer|selecionar|premiar|serão selecionados|escolhidos|contemplados|destina-se a)\b\s*)?" # Frases introdutórias opcionais
        r"(\d+)\s*(?:\((?:[a-zá-ú\s]+)\))?" # Captura o número em algarismo e, opcionalmente, o número por extenso entre parênteses
        r"\s+" # Um ou mais espaços
        r"\b(vagas|projetos|propostas|contemplados|selecionados|escolhidos|grupos)\b", # As palavras-chave
        re.IGNORECASE | re.UNICODE
    )
    return regex_vagas

def regex_verificar_porcentagem():
    """
    Verifica porcentagens (ex: 50%).
    Retorna True se uma correspondência for encontrada, False caso contrário.
    """
    regex_porcentagem = re.compile(r"\d+\s*%")
    return regex_porcentagem

def regex_cotas_negros():
    """
    Verifica termos relacionados a indivíduos e cultura negra.
    Retorna True se uma correspondência for encontrada, False caso contrário.
    """
    regex_negros = re.compile(
        r"(negros?|negras?|pessoas?\s*negras|população\s*negra|agentes?\s*culturais\s*negros?|"
        r"agentes\s*culturais\s*de\s*matriz\s*africana|culturas?\s*negras?|matriz\s*africana)",
        re.IGNORECASE
    )
    return regex_negros

def regex_cotas_indigenas():
    """
    Verifica termos relacionados a povos e cultura indígena.
    Retorna True se uma correspondência for encontrada, False caso contrário.
    """
    regex_indigenas =  re.compile(
      r"(indígenas?|povos?\s*indígenas?|agentes?\s*culturais\s*indígenas?|culturas?\s*indígenas?|agentes\s*culturais\s*indígenas?|comunidades\s*indígenas|tradições\s*indígenas|pessoas?\s*indígenas)",
      re.IGNORECASE
    )
    return regex_indigenas

def regex_cotas_pcd():
    """
    Verifica termos relacionados a pessoas com deficiência (PCD).
    Retorna True se uma correspondência for encontrada, False caso contrário.
    """
    regex_pcd =  re.compile(
        r"(pessoas?\s*com\s*deficiência|pcd|agentes?\s*culturais\s*com\s*deficiência|"
        r"agentes culturais pcd|artistas?\s*com\s*deficiência)",
        re.IGNORECASE
    )
    return regex_cotas_pcd