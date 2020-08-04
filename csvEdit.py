#Symbolism
#Impressionism
#Realism
#Expressionism

import pandas as pd

df = pd.read_csv("train_info.csv")
df = df[['filename','style']]
df=df[df['style'].isin(['Symbolism', 'Impressionism','Realism','Expressionism'])]
df.to_csv('special.csv')


'''
df_sym=df[df['style'].str.contains("Symbolism")]
df_imp=df[df['style'].str.contains("Impressionism")]
df_rea=df[df['style'].str.contains("Realism")]
df_exp=df[df['style'].str.contains("Expressionism")]

df_son=df_sym

df_son.append(df_imp)
df_son.append(df_rea)
df_son.append(df_exp)


#dt.to_csv('special.csv’)
print(df_sym)
'''