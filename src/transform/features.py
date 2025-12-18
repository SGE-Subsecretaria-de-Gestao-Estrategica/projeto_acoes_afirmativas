import pandas as pd
import numpy as np
from unicodedata import normalize

def calcula_cotas_vagas_e_valores(
    df: pd.DataFrame,
    col_valor_total: str = 'valor_total',
    col_vagas_totais: str = 'vagas_totais'
) -> pd.DataFrame:

    cotas = {
        'negras': 'perc_cotas_negras',
        'indigenas': 'perc_cotas_indigenas',
        'pcd': 'perc_cotas_pcd'
    }

    # calcula vagas
    for grupo, col_perc in cotas.items():
        df[f'vagas_cotas_{grupo}'] = (
            np.floor(df[col_perc] * df[col_vagas_totais])
            .astype('Int64')
        )

    # valor unitário
    df['valor_por_vaga'] = df[col_valor_total] / df[col_vagas_totais]

    # calcula valores
    for grupo in cotas.keys():
        df[f'valor_cotas_{grupo}'] = (
            df['valor_por_vaga'] * df[f'vagas_cotas_{grupo}']
        )

    return df


def cria_flags_cotas(df: pd.DataFrame) -> pd.DataFrame:
    regras = {
        'negras': ('perc_cotas_negras', 0.25),
        'indigenas': ('perc_cotas_indigenas', 0.10),
        'pcd': ('perc_cotas_pcd', 0.05),
    }

    for grupo, (col, limite) in regras.items():
        df[f'flag_cotas_{grupo}'] = df[col] >= limite

    return df


def cria_tipo_edital(
    df: pd.DataFrame,
    col_origem: str = 'nome_pdf_pk',
    col_destino: str = 'tipo_edital'
) -> pd.DataFrame:

    def normaliza_texto(s: str) -> str:
        s = str(s)
        s = normalize('NFD', s)
        s = s.encode('ascii', 'ignore').decode('utf-8')
        return s.lower()

    texto = df[col_origem].apply(normaliza_texto)

    sep = r'(^|[_\-\s\.])'
    end = r'([_\-\s\.]|$)'

    regras = [
        # CULTURA VIVA (prioridade máxima)
        (
            rf'{sep}(pncv|cultura{sep}?viva|ponto(s)?|pontao(s)?|pontoe?s?){end}',
            'CULTURA VIVA'
        ),

        # BOLSA
        (
            rf'{sep}bolsa(s)?{end}',
            'BOLSA'
        ),

        # SUBSÍDIO (com erro comum)
        (
            rf'{sep}(subsidio|subisidio)(s)?{end}',
            'SUBSÍDIO'
        ),

        # PRÊMIO / PREMIAÇÃO
        (
            rf'{sep}premi(o|os|acao|acoes)?{end}',
            'PRÊMIO'
        ),

        # FOMENTO (inclui QUALIFICAÇÃO)
        (
            rf'{sep}(fomento(s)?|qualificacao){end}',
            'FOMENTO'
        ),
    ]

    df[col_destino] = pd.NA

    for pattern, label in regras:
        mask = texto.str.contains(pattern, regex=True)
        df.loc[mask & df[col_destino].isna(), col_destino] = label

    return df