import pandas as pd
from transform import reshape
from config import settings
from utils import viz

# VARIÁVEIS GLOBAIS
EXECUCAO_FILE_PATH      = '../data/external/Execução Financeira - Política Nacional Aldir Blanc (1).xlsx'
ADESAO_FILE_PATH        = '../data/external/Adesão - Politica Nacional Aldir Blanc.xlsx'

IBGE_TABELA             = '../data/external/POP2024_20241230.xls'
GEO_JSON                = settings.EXTERNAL_DATA_PATH / 'br_states.json'

TABELA_SNIIC            = '../data/final/tabela_final_3_2_2026_13_53.parquet'
FINAL_TABLE_FILE_PATH   = '../data/final/tabela_final_3_2_2026_13_53.parquet'


# IMPORTS
df_execucao        = pd.read_excel(EXECUCAO_FILE_PATH)
df_adesao          = pd.read_excel(ADESAO_FILE_PATH)
df_tabela_sniic    = pd.read_parquet(FINAL_TABLE_FILE_PATH)
df_ibge_mun        = pd.read_excel(IBGE_TABELA, sheet_name='MUNICÍPIOS')
df_ibge_est        = pd.read_excel(IBGE_TABELA)


# UTILIZAMOS DF_ADESAO PARA FAZER MERGE COM A TABELA_SNIIC
df_adesao=df_adesao.rename(columns={
    'Código IBGE': 'cod_ibge',
    'UF do Ente': 'uf',
    'População': 'populacao',
    'Valor do Plano':'valor_plano'
})
df_adesao = df_adesao[['cod_ibge', 'uf']] 
df_adesao = df_adesao.drop_duplicates()
df_adesao = df_adesao.reset_index(drop=True)


def pipeline(df, tipo_ente):
    df = df.loc[df['tipo_ente'] == tipo_ente]

    # TRATAMENTO TABELA_SNIIC - CRIAMOS A TABELA DE VALORES AGREGADO
    df = reshape.agregar_tabela_sniic(df=df)

    # CALCULO DE VALOR E VAGA
    df = reshape.calcular_totais_cotas(df=df)


    # MERGE DF E DF_ADESAO
    df = df.merge(right=df_adesao, how='left', on='cod_ibge')

    if tipo_ente == "MUNICIPIO":
        # CORRIGE O RIO DE JANEIRO
        df.loc[df['ente_federativo'] == 'RIO DE JANEIRO (capital)', 'uf'] = 'RJ'
    
    # TRANSFORMA VALORES EM VAGAS
    df = reshape.calcular_proporcoes_valor_e_vagas(df=df)

    return df


df = pipeline(df=df_tabela_sniic, tipo_ente='ESTADO')
# VISUALIZAÇÃO
viz.plot_mapa_estados_continuo(
    df=df,
    geo_path=GEO_JSON,
    value_col="rel_valor_cotas_pcd",  
    modo="ESTADO",
    bins=settings.BINS_PESSOAS_PCD,
    labels=settings.LABELS_PESSOAS_PCD,
    colors=settings.COLORS_PESSOAS_PCD,
    right=True,                     
    title="Pessoas Indígenas — Relação Cotas / Valor (R$)",
    save_path="outputs/final/mapa__pessoas__pcd_260204_H1228",
    show=False
)