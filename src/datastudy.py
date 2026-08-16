
import pandas as pd
from common import get_params, normalize_text
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa
from wordcloud import WordCloud
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Muestra los diagramas mientras se calculan, pero pausa el flujo del programa hasta que se cierren
SHOW = False

# Descargar stopwords
nltk.download("stopwords", quiet=True)
nltk.download('punkt')
nltk.download('punkt_tab')

# Muestra la frecuencia por clase
def stats_nclass(df):
    cats = df['label'].unique()
    t = len(df['label'])
    print("="*10 + "Frecuencias por clase" + "="*10)
    #table = {'Clase':[],'Nº':[],'%':[]}
    for c in cats:
        n = len(df[df['label'] == c])
        #table['Clase'].append(c)
        #table['Nº'].append(n)
        #table['%'].append(n/t*100)
        print('Nº de ',c,' = ',n,' o ', f'{n/t*100:.3f}', '%')
    #table = pd.DataFrame(table)
    #plt.table(cellText=table.values,colLabels=table.columns, loc='center')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()

# Muestra ejemplos por clase de cada jurado
def stats_bars(df,pret):
    j_bars = ['a1','a2','a3','label']
    j_title = ['Jurado 1', 'Jurado 2', 'Jurado 3', 'Mayoría']
    for i,j in enumerate(j_bars):
        df[j].value_counts().plot(kind='bar', color=['skyblue','salmon'], title='Distribución de Clases de '+j_title[i],xlabel='Clase',ylabel='Nº de ejemplos')
        plt.tight_layout()
        if SHOW:
            plt.show()
        else:
            plt.savefig(pret + "bars "+j+".png")
            plt.clf()

            

# Muestra Fleiss's Kappa
def stats_fleissk(df):
    cats = df['label'].unique()
    j_cols = ['a1', 'a2', 'a3']
    c = df[j_cols].apply(lambda x: x.map({cat: i for i, cat in enumerate(cats)})).values
    mat = np.zeros((len(df), len(cats)))
    for i, row in enumerate(c):
        for val in row:
            mat[i, val] += 1

    print("="*10 + "Fleiss's Kappa" + "="*10)
    print(f"Fleiss's Kappa: {fleiss_kappa(mat):.3f}")

# Muestra matriz de confusión de juez vs mayoría
def stats_confmat(df,pret):
    j_cols = ['a1', 'a2', 'a3']
    for i,j in enumerate(j_cols):
        plt.figure(figsize=(6,4))
        sns.heatmap(pd.crosstab(df[j], df['label'], rownames=[f'Juez {i+1}'],colnames=['Mayoría'],), annot=True,fmt='g', cmap="YlGnBu")              
        plt.title(f'Juez {i+1} vs Mayoría')
        plt.tight_layout()
        
        if SHOW:
            plt.show()
        else:
            plt.savefig(pret+"mconf "+j+".png")
            plt.clf()


# Calcula valores atípicos
def _outliers(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    linf = Q1 - 1.5 * IQR
    lsup = Q3 + 1.5 * IQR
    return ((s < linf) | (s > lsup)).sum()

# Muestra diagrama de cajas y bigotes y la cantidad de valores atípicos por clase
def stats_boxplot(df,pret):
    cats = df['label'].unique()
    df['char_len'] = df['txt'].astype(str).str.len()
    #df['punct_count'] = df['txt'].astype(str).apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x))
    
    print("="*10 + "Valores atípicos por clase" + "="*10)
    for c in cats:
        n = _outliers(df[df['label'] == c]['char_len'])
        print("En", c," hay una cantidad de ", n, " ejemplos fuera del rango intercuartil")
    
    #fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x='label', y='char_len', data=df) #,ax=ax[0])
    #ax[0].set_title('Longitud de Texto')
    #sns.boxplot(x='label', y='punct_count', data=df, ax=ax[1])
    #ax[1].set_title('Cantidad de Puntuación')
    plt.tight_layout()
    if SHOW:
        plt.show()
    else:
        plt.savefig(pret+"dcabi.png")
        plt.clf()


# Muestra una nube de palabras donde las palabras más frecuentes son las más grandes
def stats_wordcloud(df,pret):
    trolling_text = " ".join(df[df['label'] == 'Trolling']['txt']).lower()
    if trolling_text:
        wc = WordCloud(background_color='white').generate(trolling_text)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Palabras comunes')
        plt.tight_layout()
        
        if SHOW:
            plt.show()
        else:
            plt.savefig(pret+"wcloud.png")
            plt.clf()

def stats_freqword(df,N,per_row):
    stop_words = set(stopwords.words("english"))

    if per_row:
        df["txt_unique"] = df["txt"].apply(lambda x: " ".join(set(str(x).lower().split())))

    for c, grp in df.groupby("label"):
        print("="*10 + str(N) + f"{" Palabras más frecuentes de "+ c:<54}" + "="*10)

        if per_row:
            txt = " ".join(grp["txt_unique"].dropna().astype(str)).lower()
        else:
            txt = " ".join(grp["txt"].dropna().astype(str)).lower()

        # Tokenización de NLTK
        tokens = word_tokenize(txt, language="english")

        # Filtrar solo palabras alfanuméricas y stopwords
        wrds = [
            word for word in tokens if word.isalnum() and word not in stop_words
        ]

        fdist = nltk.FreqDist(wrds)
        top = fdist.most_common(N)
        print(f"{'Palabra':<20} | {'Apariciones':<12} | Frecuencia")
        print("="*76)
        for w, n in top:
            print(f"{w:<20} | {n:<12} | {n/len(wrds) * 100:.3f}%")

# Muestra el nº de tokens por clase
def stat_tokencount(df,tok):
    cats = df['label'].unique()
    t = df['txt'].str.count(tok).sum()
    print("="*10 + "Frecuencia de tokens " + tok + "="*10)
    print("Nº de ",tok," Total = ", t)
    for c in cats:
     n = df[df['label'] == c]['txt'].str.count(tok).sum()
     print("Nº de ",tok," de ", c ," = ", n," o ", f"{n/t*100:.3f}" ,"%")

# Obtiene cantidad de filas duplicadas
def getdupes(df):
    return len(df['txt'])-len(df['txt'].drop_duplicates())

if __name__ == "__main__":

    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    FILE = params["DATAGEN"]["FILE"]
    STUDY_OUT = params["DATAGEN"]["STUDY_OUT"]
    ##

    sns.set_theme(style="whitegrid")

    def _statshow(filter,pret):
        # Leemos archivo
        df = pd.read_excel(FILE)
    
        # Cambios nombres de las columnas por comodidad
        df.columns = ['txt','a1','a2','a3','label']
        allowed = ["Normal","Trolling"]
        if filter:
            df = df[df['a1'].isin(allowed) & df['a2'].isin(allowed) & df['a3'].isin(allowed)] # Quitamos clases no contempladas
        # Número de ejemplos de cada clase
        stats_nclass(df)
        # Normalizamos texto
        print("="*10 + "Filas duplicadas" + "="*10)
        print("Duplicados pre-normalización: ", getdupes(df))
        df['txt'] = df['txt'].apply(lambda x: normalize_text(str(x)))
        print("Duplicados post-normalización: ", getdupes(df))

        # Contar [URL] y [USER]
        stat_tokencount(df,"[URL]")
        stat_tokencount(df,"[USER]")

        # Distribución de clases por jurado
        stats_bars(df,pret)
        # Fleiss K
        stats_fleissk(df)
        # Jurado vs Popular, matriz de confusión
        stats_confmat(df,pret)
        # Diagrama de caja y bigotes
        stats_boxplot(df,pret)
        # Frecuencia de palabras
        print("="*20 + "Frecuencia de palabras total" + "="*20)
        stats_freqword(df,20,False)
        print("")
        print("="*20 + "Frecuencia de palabras, una por fila" + "="*20)
        stats_freqword(df,20,True)

        # Word Cloud
        stats_wordcloud(df,pret)

    oout = sys.stdout
    # Guardamos salida en un archivo
    sys.stdout = open(STUDY_OUT+'/datastats.txt', 'w')

    print("="*40)
    print("Mostrando sin filtrar...")
    print("="*40)
    _statshow(False,STUDY_OUT+"/nof")

    print("\n\n")

    print("="*40)
    print("Mostrando filtrados...")
    print("="*40)
    _statshow(True,STUDY_OUT+"/sif")

    sys.stdout.close()
    sys.stdout = oout
    print("Datos analíticos guardados en: "+ STUDY_OUT)