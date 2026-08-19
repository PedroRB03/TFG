
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
from pathlib import Path
from math import inf

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

    print("\\begin{tabular}{|c|c|c|} \\hline")
    print("Clase & Ejemplares & Frecuencia \\\\ \\hline")
    for c in cats:
        n = len(df[df['label'] == c])
        print(f'{c} & {n} & {n/t*100:.2f}\\% \\\\ \\hline')
    print("\\end{tabular}")

# Muestra ejemplos por clase de cada jurado
def stats_bars(df,pret):
    j_bars = ['a1','a2','a3','label']
    j_title = ['Jurado 1', 'Jurado 2', 'Jurado 3', 'Mayoría']
    for i,j in enumerate(j_bars):
        ax = df[j].value_counts().plot(kind='bar', color=['skyblue','salmon'], title='Distribución de Clases de '+j_title[i],xlabel='Clase',ylabel='Nº de ejemplos')
        ax.margins(y=0.15)
        ax.bar_label(ax.containers[0], padding=3)
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
    c = df[j_cols].apply(lambda x: x.map({cat: i for i, cat in enumerate(cats)})).values # Reemplazamos categorías por números
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
def _outliers(df):
    Q1 = df['char_len'].quantile(0.25)
    Q3 = df['char_len'].quantile(0.75)
    IQR = Q3 - Q1
    linf = Q1 - 1.5 * IQR
    lsup = Q3 + 1.5 * IQR
    return ((df['char_len'] < linf) | (df['char_len'] > lsup))

# Muestra la cantidad de valores atípicos por clase
def stats_outliers(df):
    cats = df['label'].unique()
    df['char_len'] = df['txt'].astype(str).str.len()
    
    print("="*10 + "Valores atípicos de la longitud del texto por clase" + "="*10)

    tot = 0
    for c in cats:
        s = _outliers(df[df['label'] == c])
        outl = df[df['label'] == c][s]
        n = len(outl)
        tot+=n

    print("\\begin{tabular}{|c|c|c|} \\hline")
    print("Clase & Valores atípicos & Frecuencia \\\\ \\hline")

    for c in cats:
        s = _outliers(df[df['label'] == c])
        outl = df[df['label'] == c][s]
        n = len(outl)
        print(f'{c} & {n} & {n/tot*100:.2f}\\% \\\\ \\hline')

    print(f'Total & {tot} & \\\\ \\hline')
    print("\\end{tabular}")

# Muestra diagrama de cajas y bigotes de la longitud de los ejemplares
def stats_boxplot(df,pret,fit):
    cats = df['label'].unique()
    df['char_len'] = df['txt'].astype(str).str.len()
    
    y_min = inf
    y_max = -inf
    for c in cats:
        s = _outliers(df[df['label'] == c])
        noutl = df[df['label'] == c][s == False]
        y_min = min(y_min,noutl['char_len'].min())
        y_max = max(y_max,noutl['char_len'].max())

    margen = (y_max - y_min)* 0.05

    ax = sns.boxplot(x='label', y='char_len', data=df) #,ax=ax[0])
    ax.set_xlabel("Clase")
    ax.set_ylabel("Longitud")
    if fit:
        ax.set_ylim(y_min - margen, y_max + margen)
    plt.tight_layout()
    if SHOW:
        plt.show()
    else:
        if fit:
            plt.savefig(pret+"fitdcabi.png")
        else:
            plt.savefig(pret+"nofitdcabi.png")
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
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\begin{tabular}{|c|c|c|} \\hline")
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
        print("\\multicolumn{3}{|c|}{\\textit{"+c+"}} \\\\ \\hline")
        print("Palabra & Apariciones & Frecuencia \\\\ \\hline")
        for w, n in top:
            print(f'{w} & {n} & {n/len(wrds)*100:.2f}\\% \\\\ \\hline')
        print("\\end{tabular}")
        print("\\caption{20 palabras más frecuentes de \\textit{"+c+"}}")
        print("\\end{table}")


# Obtiene cantidad de filas duplicadas
def getdupes(df):
    return len(df['txt'])-len(df['txt'].drop_duplicates())

if __name__ == "__main__":

    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    FILE = params["DATAGEN"]["FILE"]
    STUDY_OUT = params["DATAGEN"]["STUDY_OUT"]
    ##
    Path(STUDY_OUT).mkdir(parents=True, exist_ok=True) # Creamos ruta si no existe
    
    sns.set_theme(style="whitegrid")

    def _statshow(filter,pret):
        # Leemos archivo
        df = pd.read_excel(FILE)
    
        # Cambios nombres de las columnas por comodidad
        df.columns = ['txt','a1','a2','a3','label']
        allowed = ["Normal","Trolling"]
        
        
        # Normalizamos texto
        print("="*10 + "Filas duplicadas" + "="*10)
        print("Duplicados pre-normalización: ", getdupes(df))
        df['txt'] = df['txt'].apply(lambda x: normalize_text(str(x)))
        print("Duplicados post-normalización: ", getdupes(df))

        if filter:
            df = df[df['a1'].isin(allowed) & df['a2'].isin(allowed) & df['a3'].isin(allowed)] # Quitamos clases no contempladas
            
        # Quitamos duplicados
        df = df.drop_duplicates(subset=['txt'],keep=False,ignore_index=True) 

        # Número de ejemplos de cada clase
        stats_nclass(df)

        # Distribución de clases por jurado
        stats_bars(df,pret)
        # Fleiss K
        stats_fleissk(df)
        # Jurado vs Popular, matriz de confusión
        stats_confmat(df,pret)
        # Diagrama de caja y bigotes
        stats_outliers(df)
        stats_boxplot(df,pret,fit=False)
        stats_boxplot(df,pret,fit=True)
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