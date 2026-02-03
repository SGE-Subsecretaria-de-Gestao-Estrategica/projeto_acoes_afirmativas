import pandas as pd
from transform import reshape
from config import settings
from utils import viz

# VARIÁVEIS GLOBAIS
EXECUCAO_FILE_PATH      = '../data/external/Execução Financeira - Política Nacional Aldir Blanc (1).xlsx'
ADESAO_FILE_PATH        = '../data/external/Adesão - Politica Nacional Aldir Blanc.xlsx'

IBGE_TABELA             = '../data/external/POP2024_20241230.xls'
GEO_JSON                = r'C:\Users\gabiru\Documents\GitHub\projeto_acoes_afirmativas\data\external\br_states.json'

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


# TRATAMENTO TABELA_SNIIC - CRIAMOS A TABELA DE VALORES AGREGADO
df = reshape.agregar_tabela_sniic(df=df_tabela_sniic)


# CALCULO DE VALOR E VAGA
df = reshape.calcular_totais_cotas(df=df)


# ARREDONDA PERC
cols_perc = [
    "perc_mean_cotas_negras",
    "perc_mean_cotas_indigenas",
    "perc_mean_cotas_pcd",
]
df[cols_perc] = df[cols_perc].round(2)


# MERGE DF E DF_ADESAO
df_merged = df.merge(right=df_adesao, how='left', on='cod_ibge')


# CORRIGE O RIO DE JANEIRO
df_merged.loc[df['ente_federativo'] == 'RIO DE JANEIRO (capital)', 'uf'] = 'RJ'


# CHECKPOINT
df = df_merged.copy()


# TRANSFORMA VALORES EM VAGAS
df = reshape.calcular_proporcoes_valor_e_vagas(df=df)

mask_tipo_ente = df['tipo_ente'] == 'ESTADO'

print(df.loc[mask_tipo_ente]['perc_mean_cotas_negras'].value_counts())


# VISUALIZAÇÃO
viz.plot_mapa_estados_continuo(
    df=df,
    geo_path=GEO_JSON,
    value_col="perc_mean_cotas_negras",  
    modo="ESTADO",
    bins=settings.BINS_PESSOAS_NEGRAS,
    labels=settings.LABELS_PESSOAS_NEGRAS,
    colors=settings.COLORS_PESSOAS_NEGRAS,
    right=True,                       
    title="Pessoas Negras — Percentual Médio de Cotas por Estado ",
    save_path="outputs/mapa__pessoa__negra_260302_H1532"
)
