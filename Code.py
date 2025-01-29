#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px


# In[2]:


file_path = "MICRODADOS_CADASTRO_CURSOS_2022.CSV"


try:
    cursos = pd.read_csv(file_path, encoding='latin1' , sep = ";",low_memory = False)
except UnicodeDecodeError:
    try:
        cursos = pd.read_csv(file_path, encoding='iso-8859-1', sep = ";")
    except UnicodeDecodeError:
        cursos = pd.read_csv(file_path, encoding='cp1252', sep = ";")


# In[3]:


# file_path = "MICRODADOS_CADASTRO_IES_2021.CSV"


# try:
#     ies = pd.read_csv(file_path, encoding='latin1' , sep = ";")
# except UnicodeDecodeError:
#     try:
#         ies = pd.read_csv(file_path, encoding='iso-8859-1', sep = ";")
#     except UnicodeDecodeError:
#         ies = pd.read_csv(file_path, encoding='cp1252', sep = ";")


# In[4]:


len(cursos)


# In[5]:


cursos.shape


# In[6]:


def calcular_tamanho_amostra(populacao, confianca=0.95, margem_erro=0.05, proporcao=0.5):
    from scipy.stats import norm

    # Valor Z para o nível de confiança
    z = norm.ppf(1 - (1 - confianca) / 2)

    # Fórmula para o tamanho da amostra
    numerador = populacao * z**2 * proporcao * (1 - proporcao)
    denominador = margem_erro**2 * (populacao - 1) + z**2 * proporcao * (1 - proporcao)
    tamanho_amostra = numerador / denominador

    return round(tamanho_amostra)

# Exemplo de uso
populacao = len(cursos)  # Tamanho da população
confianca = 0.95  # Nível de confiança (95%)
margem_erro = 0.05  # Margem de erro (5%)
proporcao = 0.5  # Proporção esperada (50%)

tamanho = calcular_tamanho_amostra(populacao, confianca, margem_erro, proporcao)
print(f"Tamanho necessário da amostra: {tamanho}")


# In[7]:


cursos.groupby('TP_MODALIDADE_ENSINO').count()


# In[8]:


cursos=cursos[cursos['QT_ING']>=1]


# In[9]:


cursos =cursos[cursos['QT_ING_0_17']>=1]


# In[10]:


cursos =cursos[cursos['QT_ING_18_24']>=1]


# In[11]:


cursos =cursos[cursos['QT_ING_25_29']>=1]


# In[12]:


cursos =cursos[cursos['QT_ING_30_34']>=1]


# In[13]:


cursos =cursos[cursos['QT_ING_35_39']>=1]


# In[14]:


cursos =cursos[cursos['QT_ING_40_49']>=1]


# In[15]:


cursos =cursos[cursos['QT_MAT_18_24']>=1]


# In[16]:


cursos =cursos[cursos['QT_MAT_25_29']>=1]


# In[17]:


cursos =cursos[cursos['QT_MAT_30_34']>=1]


# In[18]:


cursos =cursos[cursos['QT_MAT_35_39']>=1]


# In[19]:


cursos =cursos[cursos['QT_MAT_40_49']>=1]


# In[20]:


cursos =cursos[cursos['QT_CONC_18_24']>=1]


# In[21]:


cursos =cursos[cursos['QT_CONC_25_29']>=1]


# In[22]:


cursos =cursos[cursos['QT_CONC_30_34']>=1]


# In[23]:


cursos =cursos[cursos['QT_CONC_35_39']>=1]


# In[24]:


cursos =cursos[cursos['QT_CONC_40_49']>=1]


# In[25]:


len(cursos)


# In[26]:


cursos.groupby('TP_MODALIDADE_ENSINO').count()


# In[27]:


faixas_etarias = cursos[['QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']]


# In[28]:


ead_ou_presencial_com_faixa_etaria = cursos[['TP_MODALIDADE_ENSINO','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']]


# In[29]:


ead_ou_presencial_com_faixa_etaria


# In[30]:


ead_presencial_faixa_etaria_gratuito_privado = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']]


# In[31]:


ead_presencial_faixa_etaria_gratuito_privado


# In[32]:


ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','TP_GRAU_ACADEMICO','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']]


# In[33]:


ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico_sexo_ing = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','TP_GRAU_ACADEMICO','QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']]


# In[34]:


ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico_sexo_ing_sexo_matri_idade_matri = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','TP_GRAU_ACADEMICO','QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS','QT_MAT_FEM','QT_MAT_MASC','QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS']]


# In[35]:


ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico_sexo_ing_sexo_matri_idade_matri_sexo_conc = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','TP_GRAU_ACADEMICO','QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS','QT_MAT_FEM','QT_MAT_MASC','QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS','QT_CONC_FEM','QT_CONC_MASC']]


# In[36]:


ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico_sexo_ing_sexo_matri_idade_matri_sexo_conc_idade_conc = cursos[['TP_MODALIDADE_ENSINO','IN_GRATUITO','TP_GRAU_ACADEMICO','QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS','QT_MAT_FEM','QT_MAT_MASC','QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS','QT_CONC_FEM','QT_CONC_MASC','QT_CONC_0_17','QT_CONC_18_24','QT_CONC_25_29','QT_CONC_30_34','QT_CONC_35_39','QT_CONC_40_49','QT_CONC_50_59','QT_CONC_60_MAIS',]]


# In[37]:


df = ead_presencial_faixa_etaria_gratuito_privado_grau_acadêmico_sexo_ing_sexo_matri_idade_matri_sexo_conc_idade_conc


# In[38]:


df = df.drop(['IN_GRATUITO','TP_GRAU_ACADEMICO'],axis = 1)


# In[39]:


dados_por_modalidade = df.groupby('TP_MODALIDADE_ENSINO')[['QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS','QT_MAT_FEM','QT_MAT_MASC','QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS','QT_CONC_FEM','QT_CONC_MASC','QT_CONC_0_17','QT_CONC_18_24','QT_CONC_25_29','QT_CONC_30_34','QT_CONC_35_39','QT_CONC_40_49','QT_CONC_50_59','QT_CONC_60_MAIS']].sum()


# In[40]:


dados_por_modalidade.T


# In[41]:


somente_ingressantes = df.groupby('TP_MODALIDADE_ENSINO')[['QT_ING_FEM','QT_ING_MASC','QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']].sum()


# In[42]:


somente_ingressantes_idade = df.groupby('TP_MODALIDADE_ENSINO')[['QT_ING_0_17','QT_ING_18_24','QT_ING_25_29','QT_ING_30_34','QT_ING_35_39','QT_ING_40_49','QT_ING_50_59','QT_ING_60_MAIS']].sum()


# In[43]:


somente_ingressantes_idade = somente_ingressantes_idade.T
somente_ingressantes_idade


# In[44]:


somente_ingressantes_idade['Frequência Acumudada Presencial'] = somente_ingressantes_idade[1].cumsum()
somente_ingressantes_idade['Frequência Acumudada EAD'] = somente_ingressantes_idade[2].cumsum()
somente_ingressantes_idade['Frequência Relativa Presencial'] = somente_ingressantes_idade[1]/(somente_ingressantes_idade[1].sum())*100
somente_ingressantes_idade['Frequência Relativa EAD'] = somente_ingressantes_idade[2]/(somente_ingressantes_idade[2].sum())*100
somente_ingressantes_idade
somente_ingressantes_idade.to_csv('tabela_frequencia_ingressantes_idade.csv',sep = ';')


# In[45]:


fig =px.bar(somente_ingressantes_idade[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto= True,barmode='group')
fig


# In[46]:


somente_ingressantes_sexo = df.groupby('TP_MODALIDADE_ENSINO')[['QT_ING_FEM','QT_ING_MASC']].sum()
somente_ingressantes_sexo = somente_ingressantes_sexo.T
somente_ingressantes_sexo['Frequência Acumudada Presencial'] = somente_ingressantes_sexo[1].cumsum()
somente_ingressantes_sexo['Frequência Acumudada EAD'] = somente_ingressantes_sexo[2].cumsum()
somente_ingressantes_sexo['Frequência Relativa Presencial'] = somente_ingressantes_sexo[1]/(somente_ingressantes_sexo[1].sum())*100
somente_ingressantes_sexo['Frequência Relativa EAD'] = somente_ingressantes_sexo[2]/(somente_ingressantes_sexo[2].sum())*100
somente_ingressantes_sexo.to_csv('tabela_frequência_igressantes_sexo.csv',sep=';')


# In[47]:


px.bar(somente_ingressantes_sexo[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto= True,barmode='group')


# In[48]:


somente_matriculados = df.groupby('TP_MODALIDADE_ENSINO')[['QT_MAT_FEM','QT_MAT_MASC','QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS']].sum()


# In[49]:


somente_matriculados_idade = somente_matriculados[['QT_MAT_0_17','QT_MAT_18_24','QT_MAT_25_29','QT_MAT_30_34','QT_MAT_35_39','QT_MAT_40_49','QT_MAT_50_59','QT_MAT_60_MAIS']]


# In[50]:


somente_matriculados_idade = somente_matriculados_idade.T


# In[51]:


somente_matriculados_idade


# In[52]:


somente_matriculados_idade['Frequência Acumulada Presencial'] = somente_matriculados_idade[1].cumsum()
somente_matriculados_idade['Frequência Acumulada EAD'] = somente_matriculados_idade[2].cumsum()
somente_matriculados_idade['Frequência Relativa Presencial'] = somente_matriculados_idade[1] / (somente_matriculados_idade[1].sum()) * 100
somente_matriculados_idade['Frequência Relativa EAD'] = somente_matriculados_idade[2] / (somente_matriculados_idade[2].sum()) * 100
somente_matriculados_idade.to_csv('tabela_frequencia_matriculados_idade.csv',sep=';')


# In[53]:


px.bar(somente_matriculados_idade[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto=True,barmode='group')


# In[54]:


somente_matriculados_sexo = df.groupby('TP_MODALIDADE_ENSINO')[['QT_MAT_FEM','QT_MAT_MASC']].sum()


# In[55]:


somente_matriculados_sexo = somente_matriculados_sexo.T


# In[56]:


somente_matriculados_sexo['Frequência Acumulada Presencial'] = somente_matriculados_sexo[1].cumsum()
somente_matriculados_sexo['Frequência Acumulada EAD'] = somente_matriculados_sexo[2].cumsum()
somente_matriculados_sexo['Frequência Relativa Presencial'] = somente_matriculados_sexo[1] / (somente_matriculados_sexo[1].sum()) * 100
somente_matriculados_sexo['Frequência Relativa EAD'] = somente_matriculados_sexo[2] / (somente_matriculados_sexo[2].sum()) * 100
somente_matriculados_sexo.to_csv('tabela_frequencia_matriculados_sexo.csv')


# In[57]:


px.bar(somente_matriculados_sexo[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto = True,barmode='group')


# In[58]:


somente_concluintes = df.groupby('TP_MODALIDADE_ENSINO')[['QT_CONC_FEM','QT_CONC_MASC','QT_CONC_0_17','QT_CONC_18_24','QT_CONC_25_29','QT_CONC_30_34','QT_CONC_35_39','QT_CONC_40_49','QT_CONC_50_59','QT_CONC_60_MAIS']]


# In[59]:


somente_concluintes_idade = df.groupby('TP_MODALIDADE_ENSINO')[['QT_CONC_0_17','QT_CONC_18_24','QT_CONC_25_29','QT_CONC_30_34','QT_CONC_35_39','QT_CONC_40_49','QT_CONC_50_59','QT_CONC_60_MAIS']].sum()


# In[60]:


somente_concluintes_idade = somente_concluintes_idade


# In[61]:


somente_concluintes_idade = somente_concluintes_idade.T


# In[62]:


somente_concluintes_idade
somente_concluintes_idade['Frequência Acumulada Presencial'] = somente_concluintes_idade[1].cumsum()
somente_concluintes_idade['Frequência Acumulada EAD'] = somente_concluintes_idade[2].cumsum()
somente_concluintes_idade['Frequência Relativa Presencial'] = somente_concluintes_idade[1] / (somente_concluintes_idade[1].sum()) * 100
somente_concluintes_idade['Frequência Relativa EAD'] = somente_concluintes_idade[2] / (somente_concluintes_idade[2].sum()) * 100
somente_concluintes_idade.to_csv('tabela_frequencia_concluintes_idade.csv',sep=';')


# In[63]:


px.bar(somente_concluintes_idade[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto = True,barmode='group')


# In[64]:


somente_concluintes_sexo = df.groupby('TP_MODALIDADE_ENSINO')[['QT_CONC_FEM','QT_CONC_MASC']].sum()


# In[65]:


somente_concluintes_sexo = somente_concluintes_sexo.T


# In[66]:


somente_concluintes_sexo['Frequência Acumulada Presencial'] = somente_concluintes_sexo[1].cumsum()
somente_concluintes_sexo['Frequência Acumulada EAD'] = somente_concluintes_sexo[2].cumsum()
somente_concluintes_sexo['Frequência Relativa Presencial'] = somente_concluintes_sexo[1] / (somente_concluintes_sexo[1].sum()) * 100
somente_concluintes_sexo['Frequência Relativa EAD'] = somente_concluintes_sexo[2] / (somente_concluintes_sexo[2].sum()) * 100
somente_concluintes_sexo.to_csv('tabela_frequencia_concluintes_sexo.csv')


# In[67]:


px.bar(somente_concluintes_sexo[['Frequência Relativa Presencial','Frequência Relativa EAD']].round(2),text_auto = True,barmode='group')


# In[68]:


# Talvez, para realizar a correlação, seja necessário usar df.values


# In[69]:


len(df)


# In[70]:


df


# In[71]:


ing_presencial = cursos['QT_ING'][cursos['TP_MODALIDADE_ENSINO']==1]


# In[72]:


ing_ead = cursos['QT_ING'][cursos['TP_MODALIDADE_ENSINO']==2]


# In[73]:


conc_presencial = cursos['QT_CONC'][cursos['TP_MODALIDADE_ENSINO']==1]


# In[74]:


conc_ead = cursos['QT_CONC'][cursos['TP_MODALIDADE_ENSINO']==2]


# In[75]:


presencial = pd.DataFrame({
    'qt_ingressantes_presencial': ing_presencial,
    'qt_conc_presencial': conc_presencial
})


# In[76]:


#presencial.to_csv('presencial_corr.csv',sep = ';')


# In[77]:


ead = pd.DataFrame({
    'qt_ingressantes_ead': ing_ead,
    'qt_conc_ead': conc_ead
})


# In[78]:


ead.to_csv('ead_corr.csv',sep = ';')


# In[79]:


px.scatter(x = presencial['qt_ingressantes_presencial'],y = presencial['qt_conc_presencial'],trendline = 'ols')


# In[80]:


px.scatter(x = ead['qt_ingressantes_ead'],y = ead['qt_conc_ead'],trendline = 'ols')


# In[81]:


ead.corr()


# In[82]:


presencial.corr()


# In[83]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score


# In[84]:


X = presencial['qt_ingressantes_presencial'].values.reshape(-1,1)
y = presencial['qt_conc_presencial']


# In[85]:


X_treino,X_teste,y_treino,y_teste = train_test_split(X ,y,test_size=0.05,random_state=42 )


# In[86]:


modelo_presencial = LinearRegression()


# In[87]:


modelo_presencial.fit(X_treino,y_treino)


# In[88]:


coeficiente_angular = modelo_presencial.coef_[0]
coeficiente_linear = modelo_presencial.intercept_


# In[89]:


coeficiente_angular


# In[90]:


coeficiente_linear


# In[91]:


X2 = ead['qt_ingressantes_ead'].values.reshape(-1,1)
y2 = ead['qt_conc_ead']


# In[92]:


X2_treino,X2_teste,y2_treino,y2_teste = train_test_split(X2 ,y2,test_size=0.05, random_state=42 )


# In[93]:


modelo_ead = LinearRegression()


# In[94]:


modelo_ead.fit(X2_treino,y2_treino)


# In[95]:


coeficiente2_angular = modelo_ead.coef_[0]
coeficiente2_angular


# In[96]:


coeficiente2_linear = modelo_ead.intercept_
coeficiente2_linear

