from config import settings
from pathlib import Path
from utils import file_io
import pandas as pd
pd.set_option('display.max_columns', None)

ESTADO_FILE_NAME = Path(settings.FINAL_DATA_PATH/settings.ESTADO_V2_FILE_NAME)

if __name__=='__main__':
    df = file_io.load_to_dataframe(ESTADO_FILE_NAME)
    # RESHAPE
    df = df[settings.COLUNAS_INTERESSE_ESTADO]
    # SANITIZE
    df = df.rename(columns=RENAME_COLUNAS_ESTADO)
    # SANITIZE
    df[['perc_cotas_negras','perc_cotas_indigenas','perc_cotas_pcd']] = (
        df[['perc_cotas_negras','perc_cotas_indigenas','perc_cotas_pcd']]
        .apply(padronizar_percentual)
    )
    # SANITIZE
    df['valor_total_rs'] = moeda_br_para_float(df['valor_total'])
    # SANITIZE
    df['vagas_totais'] = df['vagas_totais'].astype('Int64')