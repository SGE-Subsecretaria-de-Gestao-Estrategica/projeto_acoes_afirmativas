from pathlib import Path

# CAMINHO DOS DIRETÓRIOS
ROOT = Path(__file__).resolve().parents[2] # PASTA RAÍZ DO PROJETO

DATA_PATH = ROOT / "data" # PASTA DE DADOS 
RAW_DATA_PATH = DATA_PATH / "raw" # RAW_DATA_PATH: onde ficam os dados crus, exatamente como chegaram da fonte, sem nenhum tratamento.
PROCESSED_DATA_PATH = DATA_PATH / "processed" # PROCESSED_DATA_PATH: onde são salvos os dados já tratados, limpos e prontos para análise/modelagem.
INTERIM_DATA_PATH = DATA_PATH / "interim" # INTERIM_DATA_PATH: área temporária para versões intermediárias durante o pipeline (não crus, mas ainda não finais).
EXTERNAL_DATA_PATH = DATA_PATH / "external" # EXTERNAL_DATA_PATH: reservatório para dados externos ao projeto (APIs, terceiros, microdados públicos, etc.).
FINAL_DATA_PATH = DATA_PATH / "final" # FINAL_DATA_PATH: entrega da obra-prima; resultados finais do pipeline prontos para consumo (dashboards, relatórios etc.).

# CAMINHOS GLOBAIS
MD_ROOT:str = Path(INTERIM_DATA_PATH/'MD')
CAPITAL_MD_ROOT:str = Path(MD_ROOT/'CAPITAL')
ESTADO_MD_ROOT:str = Path(MD_ROOT/'ESTADO')

PDF_ROOT: str = Path(RAW_DATA_PATH/'EDITAL_NOVO')
CAPITAL_PDF_ROOT: str = Path(PDF_ROOT / 'CAPITAL')
ESTADO_PDF_ROOT: str = Path(PDF_ROOT / 'ESTADO')


# CONTROLE CONVERSAO
LOG_PATH = Path(INTERIM_DATA_PATH/'controle_conversao.csv')

# ARQUIVOS DO ÚLTIMO RESULTADO
CAPITAL_FILE_NAME: str = 'output_capitais_pt2.xlsx.xlsx'
ESTADO_FILE_NAME:str = 'estados_output.xlsx'
ESTADO_V2_FILE_NAME:str = 'estados_output_v2.xlsx'
ESTADO_SHEET_NAME: str = 'dados consolidados - não mexer'

# ESTADOS
RENAME_COLUNAS_ESTADO: dict = {
    'ESTADO': 'ente_federativo',
    'pdf':'nome_pdf',
    'cotas_negras': 'perc_cotas_negras',
    'cotas_indigenas': 'perc_cotas_indigenas',
    'cotas_pcd': 'perc_cotas_pcd',
    'novo': 'is_novo'
}
COLUNAS_INTERESSE_ESTADO: list = ['ESTADO', 'pdf', 'cotas_negras', 'cotas_indigenas', 'cotas_pcd', 'vagas_totais','valor_total','novo']