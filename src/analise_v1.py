from config import settings
from pathlib import Path
from utils import file_io
import pandas as pd
import numpy as np
from transform import sanitize, features
pd.set_option('display.max_columns', None)

ESTADO_FILE_NAME  = Path(settings.FINAL_DATA_PATH/settings.ESTADO_V2_FILE_NAME)
CAPITAL_FILE_NAME = Path(settings.FINAL_DATA_PATH / settings.CAPITAL_FILE_NAME)
pd.options.display.float_format = '{:,.2f}'.format

if __name__=='__main__':
    # Carregamento da base
    df_estado  = file_io.load_to_dataframe(ESTADO_FILE_NAME)
    df_capital = file_io.load_to_dataframe(CAPITAL_FILE_NAME, sheet_name = 'dados_consolidados_nao_mudar')

    # Mantém apenas as colunas de interesse
    df_estado  = df_estado[settings.COLUNAS_INTERESSE_ESTADO]
    df_capital = df_capital[settings.CAPITAL_COLUNAS_INTERESSE]

    # Renomeia as colunas para um padrão comum
    df_estado   = df_estado.rename(columns=settings.RENAME_COLUNAS_ESTADO)
    df_capital  = df_capital.rename(columns=settings.CAPITAL_RENAME_COLUNAS)

    # Cria coluna indicativa de tipo de ente (ESTADO OU CAPITAL)
    df_estado['tipo_ente'] = 'ESTADO'
    df_capital['tipo_ente'] = 'CAPITAL'

    # Une as duas tabelas para tratamento
    df = pd.concat([df_estado, df_capital])

    # Faz a correção dos valores para percentual
    df[['perc_cotas_negras','perc_cotas_indigenas','perc_cotas_pcd']] = (
        df[['perc_cotas_negras','perc_cotas_indigenas','perc_cotas_pcd']]
        .apply(sanitize.padronizar_percentual)
    )

    # Faz correção do valor total para float
    df['valor_total'] = sanitize.moeda_br_para_float(df['valor_total'])

    # Passa vagas totais para formato integer
    df['vagas_totais'] = df['vagas_totais'].astype('Int64')

    # Corrige coluna is_novo para booleano
    df = sanitize.coluna_sim_para_bool(df, col='is_novo')

    # Cria colunas de valor por vaga para cotas
    df = features.calcula_cotas_vagas_e_valores(df)

    # Criar flag que analisa se o ente cumpriu ou não cumpriu a IN
    df = features.cria_flags_cotas(df)

    # Cria coluna de Tipo de Edital
    df = features.cria_tipo_edital(df)

    # Salva arquivos em processados 
    file_io.save_data(df, file_name='tabela_final', format='.parquet', area='final')
    file_io.save_data(df, file_name='tabela_final', format='.csv', area='final')
    file_io.save_to_excel_safe(df, file_name='tabela_final', area='final')