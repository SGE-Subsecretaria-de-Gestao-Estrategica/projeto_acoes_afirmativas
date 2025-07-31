#%%
import pandas as pd
import re, ast
# %%
path = "./../data/output/estados/df_filtrado_full.csv"
# %%
df = pd.read_csv(path)
# %%
def remove_illegal_chars(val):
    if isinstance(val, str):
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', val)
    return val


#%%
df['resultado_llm_dict'] = df['resultado_llm'].apply(ast.literal_eval)
#%%
expanded_df = pd.DataFrame(df['resultado_llm_dict'].tolist())

#%%
df_cleaned = df.drop(columns=['resultado_llm'])

# Em seguida, concatene o DataFrame limpo com as colunas expandidas
df_final = pd.concat([df_cleaned, expanded_df], axis=1)


# Aplica a função em todo o DataFrame
df_limpo = df_final.applymap(remove_illegal_chars)
#%%
# Exporta para Excel
df_limpo.to_excel("output_estados_pt.xlsx", index=False)

# %%
