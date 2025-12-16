from pathlib import Path

# CAMINHO DOS DIRETÓRIOS
ROOT = Path(__file__).resolve().parents[2] # PASTA RAÍZ DO PROJETO

DATA_PATH = ROOT / "data" # PASTA DE DADOS 
RAW_DATA_PATH = DATA_PATH / "raw" # RAW_DATA_PATH: onde ficam os dados crus, exatamente como chegaram da fonte, sem nenhum tratamento.
PROCESSED_DATA_PATH = DATA_PATH / "processed" # PROCESSED_DATA_PATH: onde são salvos os dados já tratados, limpos e prontos para análise/modelagem.
INTERIM_DATA_PATH = DATA_PATH / "interim" # INTERIM_DATA_PATH: área temporária para versões intermediárias durante o pipeline (não crus, mas ainda não finais).
EXTERNAL_DATA_PATH = DATA_PATH / "external" # EXTERNAL_DATA_PATH: reservatório para dados externos ao projeto (APIs, terceiros, microdados públicos, etc.).
FINAL_DATA_PATH = DATA_PATH / "final" # FINAL_DATA_PATH: entrega da obra-prima; resultados finais do pipeline prontos para consumo (dashboards, relatórios etc.).


# ARQUIVOS DO ÚLTIMO RESULTADO