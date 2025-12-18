from config import settings
from pathlib import Path
from utils import file_io
import pandas as pd
import numpy as np
from transform import sanitize
pd.set_option('display.max_columns', None)

ESTADO_FILE_NAME = Path(settings.FINAL_DATA_PATH/settings.ESTADO_V2_FILE_NAME)

if __name__=='__main__':
    # Carregamento da base
    df = file_io.load_to_dataframe(ESTADO_FILE_NAME)

    # Mantém apenas as colunas de interesse
    df = df[settings.COLUNAS_INTERESSE_ESTADO]

    # Renomeia as colunas para um padrão comum
    df = df.rename(columns=settings.RENAME_COLUNAS_ESTADO)

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

    # Log
    print(df.describe(include=[np.number]))
    print(df.describe(include=[object]))

    # Salva arquivos em processados 
    file_io.save_data(df, file_name='estados_processado', format='.parquet', area='processed')
    file_io.save_data(df, file_name='estados_processado', format='.csv', area='processed')