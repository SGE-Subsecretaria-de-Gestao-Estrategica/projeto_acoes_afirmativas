import pandas as pd

def classifica_tipo_ente(df: pd.DataFrame, col_cod: str = 'cod_ibge') -> pd.DataFrame:
    df = df.copy()

    cod_str = df[col_cod].astype(str)

    df['tipo_ente'] = pd.NA

    df.loc[cod_str.str.len() == 2, 'tipo_ente'] = 'ESTADO'
    df.loc[cod_str.str.len() > 2, 'tipo_ente'] = 'CAPITAL'

    return df

import numpy as np
import pandas as pd


def calcular_proporcoes_valor_e_vagas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula proporções relativas de valores financeiros e vagas destinadas às cotas.

    A função cria colunas proporcionais (razões ao total) a partir de valores
    agregados absolutos, considerando:

    - Proporção de valores de cotas em relação ao valor total agregado
      (base: ``agg_valor_total``)
    - Proporção de vagas de cotas em relação ao total de vagas agregadas
      (base: ``agg_vagas_totais``)

    As divisões só são realizadas quando o denominador é não nulo e diferente
    de zero; caso contrário, o resultado é ``NaN``.

    :param df: DataFrame contendo as colunas agregadas de valores e vagas.
    :type df: pandas.DataFrame

    :returns: O próprio DataFrame com novas colunas de proporção adicionadas.
    :rtype: pandas.DataFrame

    :raises KeyError: Se alguma das colunas necessárias não existir no DataFrame.
    """

    # --- Percentuais em VALOR (base: agg_valor_total) ---
    valor_cols = {
        "rel_valor_cotas_negras": "sum_valor_cotas_negras",
        "rel_valor_cotas_indigenas": "sum_valor_cotas_indigenas",
        "rel_valor_cotas_pcd": "sum_valor_cotas_pcd",
        "rel_valor_cotas_total": "sum_valor_cotas_total",
    }

    mask_valor = df["sum_valor_total"].notna() & (df["sum_valor_total"] != 0)

    for new_col, num_col in valor_cols.items():
        df[new_col] = np.where(
            mask_valor,
            df[num_col] / df["sum_valor_total"],
            np.nan,
        )

    # --- Percentuais em VAGAS (base: sum_vagas_totais) ---
    vagas_cols = {
        "rel_vagas_cotas_negras": "sum_vagas_cotas_negras",
        "rel_vagas_cotas_indigenas": "sum_vagas_cotas_indigenas",
        "rel_vagas_cotas_pcd": "sum_vagas_cotas_pcd",
        "rel_vagas_cotas_totais": "sum_vagas_cotas_total",
    }

    mask_vagas = df["sum_vagas_totais"].notna() & (df["sum_vagas_totais"] != 0)

    for new_col, num_col in vagas_cols.items():
        df[new_col] = np.where(
            mask_vagas,
            df[num_col] / df["sum_vagas_totais"],
            np.nan,
        )

    return df

def calcular_totais_cotas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os totais agregados de valores e vagas destinados às cotas.

    A função cria as colunas:
    - ``agg_valor_cotas_total``: soma dos valores destinados às cotas
      (negras, indígenas e PCD), tratando ausências como zero.
    - ``agg_vagas_cotas_total``: soma das vagas destinadas às cotas
      (negras, indígenas e PCD), tratando ausências como zero.

    :param df: DataFrame contendo as colunas agregadas de valores e vagas por cota.
    :type df: pandas.DataFrame

    :returns: O próprio DataFrame com as colunas de totais adicionadas.
    :rtype: pandas.DataFrame

    :raises KeyError: Se alguma das colunas necessárias não existir no DataFrame.
    """

    df["sum_valor_cotas_total"] = (
        df["sum_valor_cotas_negras"].fillna(0)
        + df["sum_valor_cotas_indigenas"].fillna(0)
        + df["sum_valor_cotas_pcd"].fillna(0)
    )

    df["sum_vagas_cotas_total"] = (
        df["sum_vagas_cotas_negras"].fillna(0)
        + df["sum_vagas_cotas_indigenas"].fillna(0)
        + df["sum_vagas_cotas_pcd"].fillna(0)
    )

    return df


def agregar_tabela_sniic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a tabela SNIIC por ente federativo, código IBGE e tipo de ente.

    A função realiza operações de soma e média sobre as métricas de valores,
    vagas e percentuais de cotas, produzindo uma tabela agregada no nível de
    análise do ente federativo.

    Agregações realizadas:
      - Somas de valores financeiros
      - Somas de vagas totais e por categoria de cota
      - Médias dos percentuais de cotas

    :param df: DataFrame SNIIC no nível original de observação.
    :type df: pandas.DataFrame

    :returns: DataFrame agregado por ``ente_federativo``, ``cod_ibge`` e
              ``tipo_ente``.
    :rtype: pandas.DataFrame

    :raises KeyError: Se alguma das colunas necessárias não existir no DataFrame.
    """

    df_agg = (
        df.groupby(
            ["cod_ibge", "ente_federativo", "tipo_ente"],
            as_index=False,
        )
        .agg(
            sum_valor_total=("valor_total", "sum"),
            sum_valor_cotas_negras=("valor_cotas_negras", "sum"),
            sum_valor_cotas_indigenas=("valor_cotas_indigenas", "sum"),
            sum_valor_cotas_pcd=("valor_cotas_pcd", "sum"),
            sum_vagas_totais=("vagas_totais", "sum"),
            sum_vagas_cotas_negras=("vagas_cotas_negras", "sum"),
            sum_vagas_cotas_indigenas=("vagas_cotas_indigenas", "sum"),
            sum_vagas_cotas_pcd=("vagas_cotas_pcd", "sum"),
            # perc_mean_cotas_negras=("perc_cotas_negras", "mean"),
            # perc_mean_cotas_indigenas=("perc_cotas_indigenas", "mean"),
            # perc_mean_cotas_pcd=("perc_cotas_pcd", "mean"),
        )
    )

    return df_agg