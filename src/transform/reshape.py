import pandas as pd

def classifica_tipo_ente(df: pd.DataFrame, col_cod: str = 'cod_ibge') -> pd.DataFrame:
    df = df.copy()

    cod_str = df[col_cod].astype(str)

    df['tipo_ente'] = pd.NA

    df.loc[cod_str.str.len() == 2, 'tipo_ente'] = 'ESTADO'
    df.loc[cod_str.str.len() > 2, 'tipo_ente'] = 'CAPITAL'

    return df