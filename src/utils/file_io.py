import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Literal, Union
from config import settings
import os
import zipfile 
import tempfile
import shutil 


AreaType = Literal["raw", "interim", "processed", "external", "final"]

_PATH_MAP = {
    "raw": settings.RAW_DATA_PATH,
    "interim": settings.INTERIM_DATA_PATH,
    "processed": settings.PROCESSED_DATA_PATH,
    "external": settings.EXTERNAL_DATA_PATH,
    "final": settings.FINAL_DATA_PATH,
}


def today() -> str:
    """Retorna timestamp curto para nome de arquivo."""
    t = datetime.today()
    return f"{t.day}_{t.month}_{t.year}_{t.hour}_{t.minute}"


def create_path_file(
    file_name: str,
    area: AreaType,
    format: str = ".csv"
) -> Path:
    """Gera caminho completo com diretório e timestamp."""
    directory = _PATH_MAP[area]
    directory.mkdir(parents=True, exist_ok=True)
    file_name_with_ext = f"{file_name}_{today()}{format}"
    return directory / file_name_with_ext


def save_data(
    dataframe: pd.DataFrame,
    file_name: str,
    format: str = ".csv",
    area: AreaType = "interim",
) -> Path:
    """Salva DataFrame em CSV, PKL ou PARQUET com timestamp e retorno do caminho.
    AreaType = ["raw", "interim", "processed", "external", "final"]
    """
    file_path = create_path_file(file_name=file_name, area=area, format=format)

    if format == ".csv":
        dataframe.to_csv(file_path, sep=';', index=False)
    elif format == ".pkl":
        dataframe.to_pickle(file_path)
    elif format == ".parquet":
        dataframe.to_parquet(file_path)
    else:
        raise ValueError("Formato não suportado. Use '.csv','.pkl' ou '.parquet'.")

    return file_path


def save_to_excel_safe(
    dataframe: pd.DataFrame,
    file_name: str,
    area: AreaType = "interim",
) -> Path:
    """
    Salva DataFrame em Excel sem o Excel alterar zeros à esquerda, formatos ou datas.
    Força colunas do tipo objeto a serem salvas como texto.
    """
    file_path = create_path_file(file_name=file_name, area=area, format=".xlsx")

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="data")
        workbook = writer.book
        worksheet = writer.sheets["data"]

        text_fmt = workbook.add_format({"num_format": "@", "text_wrap": False})

        for idx, dtype in enumerate(dataframe.dtypes):
            if dtype == object:
                worksheet.set_column(idx, idx, None, text_fmt)

    return 


def load_to_dataframe(file_path: Union[str, Path], sheet_name: str = None, dtype: bool = False) -> pd.DataFrame:
    """
    Carrega diversos formatos de arquivo para um DataFrame com base na extensão.
    
    Parâmetros
    ----------
    file_path : str | Path
        Caminho do arquivo a ser carregado. Pode ser passado como string ou Path.

    Retorno
    -------
    pd.DataFrame
        DataFrame resultante do carregamento do arquivo.

    Extensões suportadas
    --------------------
    .csv, .xlsx, .xls, .parquet, .json, .pkl, .pickle, .txt
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in [".csv"]:
        if dtype is False:
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(file_path, sep=';', dtype=str)

    elif ext in [".xlsx", ".xls"]:
        if sheet_name is None:
            return pd.read_excel(file_path, dtype=str)
        else:
            return pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)

    elif ext in [".parquet"]:
        return pd.read_parquet(file_path, engine="fastparquet")

    elif ext in [".json"]:
        return pd.read_json(file_path)

    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)

    elif ext in [".txt"]:
        # Assume arquivo delimitado por ; ou ,
        try:
            return pd.read_csv(file_path, sep=";", dtype=str)
        except Exception:
            return pd.read_csv(file_path, sep=",", dtype=str)
    
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")


def json_to_dataframe(json: list) -> pd.DataFrame: # Passar para utils depois
    """Normaliza dados dos Municípios para DataFrame.
        Será usada a sessão de região imediata possui possui 1 missing a menos
        do que a sessão de microrregão, para uf_id, sigla_uf e uf"""
    df = pd.json_normalize(json)
    return df


def ler_pasta_zip_em_dataframe(pasta_zip: str | Path) -> pd.DataFrame:
    """
    Lê arquivos .csv contidos dentro de múltiplos .zip do CNEFE (IBGE),
    concatenando todos em um único DataFrame Pandas.

    Parâmetros
    ----------
    pasta_zip : str | Path
        Caminho da pasta contendo os arquivos .zip do CNEFE.

    Retorno
    -------
    pd.DataFrame
        DataFrame com todos os registros concatenados.
    """
    
    pasta_zip = Path(pasta_zip)
    dfs = []
    
    for arquivo in pasta_zip.glob("*.zip"):
        print(f"📦 Processando {arquivo.name}...")
        with zipfile.ZipFile(arquivo, "r") as z:
            for nome in z.namelist():
                if nome.lower().endswith(".csv"):
                    df_temp = pd.read_csv(
                        z.open(nome),
                        sep=";",
                        low_memory=False,
                        dtype=str,
                        encoding="latin-1"
                    )
                    dfs.append(df_temp)
                    break  # garante que lê apenas 1 CSV por ZIP
    
    if not dfs:
        raise ValueError("Nenhum arquivo CSV foi encontrado dentro dos ZIPs.")

    return pd.concat(dfs, ignore_index=True)


def processar_zip_para_parquet(pasta_zip: str | Path, destino_parquet: str | Path, chunksize: int = 200_000):    
    pasta_zip = Path(pasta_zip)
    destino_parquet = Path(destino_parquet)
    destino_parquet.mkdir(exist_ok=True)

    for arquivo in pasta_zip.glob("*.zip"):
        print(f"⚙️ Processando {arquivo.name}...")
        contador = 0
        
        with zipfile.ZipFile(arquivo, "r") as z:
            for nome in z.namelist():
                if nome.lower().endswith(".csv"):
                    
                    for chunk in pd.read_csv(
                        z.open(nome),
                        sep=";",
                        encoding="latin-1",
                        low_memory=False,
                        chunksize=chunksize
                    ):
                        # 🧠 1) Padronizar colunas VAL_COMP_* como texto
                        colunas_val = [c for c in chunk.columns if c.startswith("VAL_COMP_")]
                        chunk[colunas_val] = chunk[colunas_val].astype(str)

                        # 🧠 2) Garantir texto em campos que podem misturar tipos
                        for col in ["NUMERO", "NUM_IMOVEL", "COMPL"]:
                            if col in chunk.columns:
                                chunk[col] = chunk[col].astype(str)

                        # 🧠 3) Código IBGE / setor como STRING (evita perder zeros)
                        for col in ["COD_MUN", "COD_SETOR"]:
                            if col in chunk.columns:
                                chunk[col] = chunk[col].astype(str)

                        # 💾 Salvar em parquet padronizado
                        nome_parquet = destino_parquet / f"{arquivo.stem}_{contador:05}.parquet"
                        chunk.to_parquet(nome_parquet, index=False)
                        contador += 1
                    break

    print("🎉 Finalizado! Todos os Parquets salvos com schema padronizado.")


def extrair_zip_cnefe_para_csv(pasta_zip: str | Path, pasta_csv: str | Path):
    pasta_zip = Path(pasta_zip)
    pasta_csv = Path(pasta_csv)
    pasta_csv.mkdir(exist_ok=True, parents=True)

    for arquivo in pasta_zip.glob("*.zip"):
        destino = pasta_csv / f"{arquivo.stem}.csv"
        if destino.exists():
            print(f"⏩ pulando {arquivo.name} (já extraído)")
            continue

        print(f"📦 extraindo {arquivo.name} ...")
        with zipfile.ZipFile(arquivo) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with open(destino, "wb") as f:
                f.write(z.read(csv_name))  # 👈 sem carregar para RAM

    print(f"🎉 Extração concluída! CSVs em: {pasta_csv}")


def consolidar_csv_cnefe_para_parquet(pasta_csv: str | Path, destino_parquet: str | Path, chunk_size: int = 1_000_000):
    pasta_csv = Path(pasta_csv)
    destino_parquet = Path(destino_parquet)

    # apagar o parquet caso já exista
    if destino_parquet.exists():
        destino_parquet.unlink()

    for csv in sorted(pasta_csv.glob("*.csv")):
        print(f"📌 processando {csv.name} ...")

        # streaming do csv
        stream = pl.read_csv_batched(
            csv,
            separator=";",
            encoding="latin-1",
            try_parse_dates=False,
            infer_schema_length=0,
            batch_size=chunk_size
        )

        # leitura lote a lote (chunk a chunk)
        while True:
            batch = stream.next_batches(1)  # pega 1 batch
            if not batch:
                break

            lf = (
                batch[0]
                .lazy()
            )

            # append no parquet
            lf.collect().write_parquet(
                destino_parquet,
                compression="zstd",
                append=True
            )

    print(f"🎯 Parquet final gerado em {destino_parquet}")

def csv_para_parquet_em_partes(pasta_csv: str | Path, pasta_parquet: str | Path, chunk: int = 800_000):
    pasta_csv = Path(pasta_csv)
    pasta_parquet = Path(pasta_parquet)

    pasta_parquet.mkdir(exist_ok=True, parents=True)

    for csv in sorted(pasta_csv.glob("*.csv")):
        uf = csv.stem  # ex: 35_SP
        print(f"📌 processando {uf} ...")

        stream = pl.read_csv_batched(
            csv,
            separator=";",
            encoding="latin-1",
            infer_schema_length=0,
            batch_size=chunk
        )

        i = 0
        while True:
            batch = stream.next_batches(1)
            if not batch:
                break

            df = (
                batch[0]
                .lazy()
                .collect()
            )

            fname = pasta_parquet / f"{uf}_{i:03d}.parquet"
            df.write_parquet(fname, compression="zstd")
            print(f"   ➕ chunk {i} → {fname.name}")
            i += 1

    print(f"🎉 Parquets em partes concluídos em: {pasta_parquet}")


def unir_parquets(pasta_parquet: str | Path, destino: str | Path):
    pasta_parquet = Path(pasta_parquet)
    destino = Path(destino)

    if destino.exists():
        destino.unlink()

    print("🔗 Unindo todos os parquet (lazy, sem RAM)...")

    df = pl.scan_parquet(f"{pasta_parquet}/*.parquet")

    df.sink_parquet(destino, compression="zstd")
    print(f"🎯 Arquivo final gerado em: {destino}")



def _extrair_csv_para_temp(z, nome):
    """Extrai um CSV de dentro do ZIP para arquivo temporário e retorna seu path."""
    temp_file = Path(tempfile.mktemp(suffix=".csv"))
    with open(temp_file, "wb") as f:
        f.write(z.read(nome))  # 👈 sem leitura inteira para RAM, grava direto
    return temp_file

def consolidar_cnefe_zip_para_parquet(
    pasta_zip: str | Path,
    destino_parquet: str | Path,
    compression: str = "zstd",
    verbose: bool = True
):
    pasta_zip = Path(pasta_zip)
    destino_parquet = Path(destino_parquet)

    lazy_frames = []

    for arquivo in pasta_zip.glob("*.zip"):
        if verbose:
            print(f"📦 lendo {arquivo.name} ...")

        with zipfile.ZipFile(arquivo) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))

            # 🥇 PRIMEIRA TENTATIVA: ler header e CSV direto do ZIP
            try:
                # HEADER direto do ZIP
                header = pl.read_csv(
                    z.open(csv_name),
                    separator=";",
                    encoding="latin-1",
                    infer_schema_length=0,
                    ignore_errors=True,
                    n_rows=0
                ).columns

                schema = {c: pl.Utf8 for c in header}

                # CSV completo direto do ZIP
                lf = (
                    pl.read_csv(
                        z.open(csv_name),
                        separator=";",
                        encoding="latin-1",
                        schema_overrides=schema
                    )
                    .lazy()
                    .drop(["DSC_MODIFICADOR", "LATITUDE", "LONGITUDE"])
                )

            except Exception:
                # 🧠 ZIP gigante (ex: SP, MG): extrair SEM RAM
                if verbose:
                    print(f"💾 extraindo para disco: {arquivo.name} ...")

                csv_temp = _extrair_csv_para_temp(z, csv_name)

                # HEADER agora pelo arquivo físico
                header = pl.read_csv(
                    csv_temp,
                    separator=";",
                    encoding="latin-1",
                    infer_schema_length=0,
                    ignore_errors=True,
                    n_rows=0
                ).columns

                schema = {c: pl.Utf8 for c in header}

                # CSV completo pelo arquivo físico, sem ZIP
                lf = (
                    pl.read_csv(
                        csv_temp,
                        separator=";",
                        encoding="latin-1",
                        schema_overrides=schema
                    )
                    .lazy()
                    .drop(["DSC_MODIFICADOR", "LATITUDE", "LONGITUDE"])
                )

            lazy_frames.append(lf)

    # 🔁 Concatenar lazy (sem RAM)
    df = pl.concat(lazy_frames, how="diagonal")

    # 💾 Salvar em parquet final
    df.sink_parquet(destino_parquet, compression=compression)

    if verbose:
        print(f"🎉 Arquivo gerado em: {destino_parquet}")

    return destino_parquet