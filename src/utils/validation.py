from pathlib import Path
import pandas as pd

def contar_editais_pdf(base_path: Path) -> int:
    """
    Conta arquivos PDF recursivamente dentro de um diretório.
    """
    return len(list(base_path.rglob("*.pdf")))


def contar_editais_md(base_path: Path) -> int:
    """
    Conta arquivos PDF recursivamente dentro de um diretório.
    """
    return len(list(base_path.rglob("*.md")))

def resumo_nans(df: pd.DataFrame) -> pd.DataFrame:
    total_linhas = len(df)

    resumo = (
        df.isna()
          .sum()
          .to_frame(name='qtd_nan')
    )

    resumo['perc_nan'] = (resumo['qtd_nan'] / total_linhas) * 100

    return resumo