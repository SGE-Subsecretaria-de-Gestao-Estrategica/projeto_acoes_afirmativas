from utils import file_io, validation
from transform import features, sanitize, reshape
import pandas as pd
from config import settings
from pathlib import Path
pd.set_option('display.max_columns', None)

# DATA_PRINCIPAL_FILE_PATH = '../data/processed/Ações Afirmativas na Política Nacional Aldir Blanc - Análise de Descumprimentos.xlsx'
# DATA_V2                  = '../data/processed/base_dados_09_01_26_14h50.xlsx'
DATA_V3                  = '../data/processed/base_dados_09_01_26_19h50.xlsx'
# DATA_V2_ABA              = 'base_dados_09-01-26'
DATA_V3_ABA              = 'base_dados_09_01_26_completa'
GEO_DATA_EST             = 'ibge_est_regioes_5_12_2025_14_49.parquet'
PLANILHA_ADESAO          = '../data/external/Adesão - Politica Nacional Aldir Blanc.xlsx'

# df = pd.read_excel(DATA_PRINCIPAL_FILE_PATH, sheet_name='Resultado Final')
df_v2 = pd.read_excel(DATA_V3, sheet_name=DATA_V3_ABA)

def transform_deprected():
    # Exclui a coluna de parecer
    df = df.drop(columns=['Conferência/Parecer'])
    df = df.reset_index(drop=True)

    # Recalcula as flags
    df = features.cria_flags_cotas(df)

    # Existe apenas uma linha que possui valor vazio para 'tipo_ente'. Trata-se do Maranhão - aqui faço essa correção
    df.loc[df['tipo_ente'].isna(), 'tipo_ente'] = 'ESTADO'

    

    # Salva em excel
    file_io.save_to_excel_safe(df, file_name='tabela_final', area='final')

    return df


def transform_v2():
    df = df_v2
    # Atribui código IBGE para o ente
    df["ente_norm"] = df["ente_federativo"].apply(sanitize.normalizar_texto)
    df["cod_ibge"]  = df["ente_norm"].apply(features.resolve_cod_ibge)
    df              = df.drop(columns=['ente_norm'])

    # Cria a coluna 'tipo_ente', discriminando os entes em 'ESTADO'(349) e 'CAPITAL'(147)
    df = reshape.classifica_tipo_ente(df)

    # Passa vagas totais para formato integer
    df['vagas_totais'] = df['vagas_totais'].astype('Int64')

    # Cria colunas de valor por vaga para cotas
    df = features.calcula_cotas_vagas_e_valores(df, type_round='round')
    # Faz o mesmo para o caso daquelas que possuem vagas vazias
    df = features.calcula_valor_cotas_sem_vagas(df)

    # Cria coluna de Tipo de Edital
    df = features.cria_tipo_edital(df)

    print(df['exclusivo'].value_counts())
    tipo_exclusivo = 'PCD'



    print(f"Tamanho Base ESTADO- {tipo_exclusivo}")
    print(len(df[(df['tipo_ente'] == 'ESTADO')&(df['exclusivo']==tipo_exclusivo)]))
    print(f"Tamanho Base CPITAL- {tipo_exclusivo}")
    print(len(df[(df['tipo_ente'] == 'CAPITAL')&(df['exclusivo']==tipo_exclusivo)]))
    print(f"Vagas total ESTADOS - {tipo_exclusivo}")
    print(df[(df['tipo_ente'] == 'ESTADO')&(df['exclusivo']==tipo_exclusivo)]['vagas_totais'].sum())
    print(f"Vagas total CAPITAIS - {tipo_exclusivo}")
    print(df[(df['tipo_ente'] == 'CAPITAL')&(df['exclusivo']==tipo_exclusivo)]['vagas_totais'].sum())

    print(f"\nValor total ESTADOS - {tipo_exclusivo}")
    print(df[(df['tipo_ente'] == 'ESTADO')&(df['exclusivo']==tipo_exclusivo)]['valor_total'].sum())
    print(f"Valor total CAPITAIS - {tipo_exclusivo}")
    print(df[(df['tipo_ente'] == 'CAPITAL')&(df['exclusivo']==tipo_exclusivo)]['valor_total'].sum())

    # features.print_resumo(df)
    # print('++++++++++++++++++++')
    # print('\n++++++++++++++++++++')
    # print('\n++++++++++++++++++++')
    # features.print_resumo(df, categoria_cota='cotas_indigenas')
    # print('++++++++++++++++++++')
    # print('\n++++++++++++++++++++')
    # print('\n++++++++++++++++++++')
    # features.print_resumo(df, categoria_cota='cotas_pcd')


    file_io.save_data(df, file_name='tabela_final', format='.parquet', area='final')
    file_io.save_data(df, file_name='tabela_final', format='.csv', area='final')
    file_io.save_to_excel_safe(df, file_name='tabela_final', area='final')

def planilha_adesao():
    df_adesao = pd.read_excel(PLANILHA_ADESAO)
    print(df_adesao.head(5))
    print(df_adesao.columns)
    print(df_adesao.dtypes)


if __name__=='__main__':
    transform_v2()
