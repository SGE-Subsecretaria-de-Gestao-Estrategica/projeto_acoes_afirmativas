import pandas as pd
import numpy as np
import re

def padronizar_percentual(serie: pd.Series) -> pd.Series:
    def converter(valor):
        if pd.isna(valor):
            return np.nan

        valor_str = str(valor).strip()

        # Caso com %
        if "%" in valor_str:
            try:
                return float(valor_str.replace("%", "").replace(",", ".")) / 100
            except ValueError:
                return np.nan

        # Caso numérico
        try:
            num = float(valor_str.replace(",", "."))
        except ValueError:
            return np.nan

        # Se for maior que 1, interpretamos como percentual inteiro
        if num > 1:
            return num / 100

        # Caso contrário, já é fração
        return num

    return serie.apply(converter).astype(float)

def moeda_br_para_float(serie: pd.Series) -> pd.Series:
    def converter(valor):
        if pd.isna(valor):
            return np.nan

        # Se já for número, retorna direto
        if isinstance(valor, (int, float)):
            return float(valor)

        valor_str = str(valor).strip()

        if valor_str == "":
            return np.nan

        # Remove símbolos (R$, espaços etc.)
        valor_str = re.sub(r"[^\d,\.]", "", valor_str)

        # Caso típico BR: tem vírgula decimal
        if "," in valor_str:
            valor_str = valor_str.replace(".", "")
            valor_str = valor_str.replace(",", ".")
            return float(valor_str)

        # Caso sem vírgula:
        # assume que já está no padrão correto
        return float(valor_str)

    return serie.apply(converter)