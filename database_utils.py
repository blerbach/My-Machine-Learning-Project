import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest, spearmanr, mannwhitneyu, kruskal

def GetIdeb(url):
    '''
    Esta funcao realiza o download da base de IDEB acessando o site do INEP
    Parametros:
    url: url onde consta disponibilizada o arquivo original
    Retorno:
    df: dataframe contendo as informacoes do IDEB
    '''
    try:
        r = requests.get(url, allow_redirects=True)

        f = ZipFile(BytesIO(r.content))
        df = f.read('divulgacao_ensino_medio-escolas-2017.xlsx')

        df = pd.read_excel(BytesIO(df), skiprows=6)

        print("Download do arquivo IDEB concluído")

    except Exception as e:
        raise print("Erro no download dos dados do IDEB. Foi verificada a exceção: " + str(e))
    
    return df

def PrepareIdeb(df):
    '''
    Essa funcao realiza o tratamento da base do IDEB 
    Parametros:
    df: dataframe baixado do site do INEP
    Retorno:
    ideb: dataframe contendo as informacoes do IDEB apos tratativa
    '''    
    try:
        ideb = df[['Código da Escola', 'Unnamed: 12', 'Unnamed: 15', 'IDEB\n2017\n(N x P)']]
        
        ideb.rename(columns={'Código da Escola' : 'CO_ESCOLA',
                            'Unnamed: 12' : 'IN_RENDIMENTO',
                            'Unnamed: 15' : 'NT_PADRONIZADA',
                            'IDEB\n2017\n(N x P)' : 'IDEB'}, inplace=True)

        ideb = ideb[ideb['CO_ESCOLA'].notna()]
        ideb[['IDEB']] = ideb[['IDEB']].replace({'-': np.nan})
        ideb[['IN_RENDIMENTO']] = ideb[['IN_RENDIMENTO']].replace({'-': np.nan})
        ideb[['NT_PADRONIZADA']] = ideb[['NT_PADRONIZADA']].replace({'-': np.nan})
        ideb['CO_ESCOLA'] = ideb['CO_ESCOLA'].astype(str).str[:-2].astype(object)

        print('Pré-processamento da base do IDEB concluído')
    except Exception as e:
        raise print('Erro na preparação dos dados do IDEB. Foi verificada a exceção: ' + str(e))

    return ideb

def GetMerged(enem, ideb):
    '''
    Essa funcao realiza o merge das tabelas do IDEB e ENEM
    Elimina variaveis identificadas com alto percentual de dados faltantes e aplica metodos de imputacao em outras 
    Parametros:
    enem: dataframe do ENEM
    ideb: dataframe do IDEB
    Retorno:
    dfmerge: dataframe contendo a tratativa
    '''
    try:
        ideb['CO_ESCOLA'] = ideb['CO_ESCOLA'].astype(str)

        ideb.sort_values(by=['CO_ESCOLA'], inplace=True)
        enem.sort_values(by=['CO_ESCOLA'], inplace=True)

        dfmerge = enem.merge(ideb, left_on='CO_ESCOLA', right_on='CO_ESCOLA', how='left')
        dfmerge.drop(labels=['CO_ESCOLA'], axis=1, inplace=True)

        # Descartando devido ao alto percentual de dados faltantes
        dfmerge.drop(labels=['IDEB', 'NT_PADRONIZADA'], axis=1, inplace=True)

        # As idades faltantes serão substituídas pela idade mediana devido à existência de idades discrepantes
        dfmerge['NU_IDADE'] = dfmerge['NU_IDADE'].astype('float64')
        dfmerge['NU_IDADE'].fillna(dfmerge['NU_IDADE'].median(), inplace=True)

        # Por se tratar de uma variável nominal, o estado civil e o tipo de ensino serão substituídos pela moda observada (Solteiro e Ensino Regular)
        dfmerge['TP_ESTADO_CIVIL'].fillna(dfmerge['TP_ESTADO_CIVIL'].mode()[0], inplace=True)
        dfmerge['TP_ENSINO'].fillna(dfmerge['TP_ENSINO'].mode()[0], inplace=True)

        # As taxas de rendimento faltantes serão substituídas pela taxa de rendimento mediana da UF de residência daquele aluno
        dfmerge['IN_RENDIMENTO'] = dfmerge['IN_RENDIMENTO'].fillna(dfmerge.groupby('SG_UF_RESIDENCIA')['IN_RENDIMENTO'].transform('median'))
        
        dfmerge = dfmerge.applymap(str)
        cols = ['NU_IDADE', 'NU_NOTA_TOT', 'IN_RENDIMENTO', 'Q005']
        for col in cols:
                dfmerge[col] = dfmerge[col].apply(float)
                
        print('Gerador de massa de dados concluído com sucesso')
    except Exception as e:
        raise print("Erro no gerador de massa de dados. Foi verificada a exceção: " + str(e))
    return dfmerge

def NormalityTest(df, alpha = 0.05):
    '''
    Esta funcao retorna os resultados dos testes de normalidade para a distribuicao de cada campo existente no dataframe indicado de acordo com o nivel de significancia escolhido
    Parametros:
    df: dataframe original
    alpha: nivel de significancia (default = 0.05)
    Retorno: 
    output: dataframe contendo os resultados dos testes de hipotese para cada variavel numerica
    '''
    input = df.select_dtypes(include = ['float64', 'int64'])
    output_1 = []
    output_2 = []
    # Shapiro Wilk
    x = 0
    for x in range(input.shape[1]):
        stat, p = shapiro(input.iloc[:, x])   
        if p > alpha:
            msg = 'Amostra tem distribuição gaussiana (H0 não foi rejeitada)'
        else:
            msg = 'Amostra não possui distribuição gaussiana (Rejeita-se H0)'
        output_1.append([input.iloc[:, x].name, msg])
    output_1 = pd.DataFrame(output_1, columns = ['Variável', 'Shapiro Wilk Result'])
    # D'Agostino-Pearson
    x = 0
    for x in range(input.shape[1]):
        stat, p = normaltest(input.iloc[:, x])   
        if p > alpha:
            msg = 'Amostra tem distribuição gaussiana (H0 não foi rejeitada)'
        else:
            msg = 'Amostra não possui distribuição gaussiana (Rejeita-se H0)'
        output_2.append([input.iloc[:, x].name, msg])
    output_2 = pd.DataFrame(output_2, columns = ['Variável', 'D Agostinos K2 Result'])
    output = output_1.merge(output_2, on='Variável', how='inner')
    return output

def NonParamTest(df, col, alpha = 0.05):
    '''
    Esta funcao retorna os resultados dos testes nao-parametricos de Mann-Whitney ou Kruskal-Wallis
    Parametros:
    df: dataframe original
    alpha: nivel de significancia (default = 0.05)
    Retorno: print dos resultados dos testes de hipotese + valor-p + variavel testada
    '''    
    # Seleciona as variável que irá ser avaliada de acordo com o target
    df_ = df[['NU_NOTA_TOT', col]]
    
    # k é o número de níveis da variável categórica col selecionada
    k = df_[col].nunique()
    
    # Temos um total de k amostras independentes 
    samples = []
    for i in range(k):
        cat = df_[col].unique()[i]
        sample = df_[df_[col] == cat].drop(col, axis=1)
        sample = np.array(sample) 
        samples.append(sample)
        
    # Comparando as distribuições
    if k == 2:
        stat, p = mannwhitneyu(samples[0], samples[1])
    elif k  == 3:
        stat, p = kruskal(samples[0], samples[1], samples[2])
    elif k  == 4:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3])
    elif k  == 5:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4])
    elif k  == 6:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4], samples[5])
    elif k  == 7:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4], samples[5], samples[6])
    elif k  == 8:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4], samples[5], samples[6], samples[7])
    elif k  == 17:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4], samples[5], samples[6], samples[7],
                          samples[8], samples[9], samples[10], samples[11], samples[12], samples[13], samples[14], samples[15], 
                          samples[16],  alternative='two-sided')        
    elif k  == 27:
        stat, p = kruskal(samples[0], samples[1], samples[2], samples[3], samples[4], samples[5], samples[6], samples[7],
                          samples[8], samples[9], samples[10], samples[11], samples[12], samples[13], samples[14], samples[15], 
                          samples[16], samples[17], samples[18], samples[19], samples[20], samples[21], samples[22], samples[23], 
                          samples[24], samples[25], samples[26], alternative='two-sided')        
        
    # Interpretando
    if p > alpha:
        print('Mesma distribuição (H0 não foi rejeitada)', p, col) 
    else:
        print('Distribuições diferentes (Rejeita-se H0)', p, col)