import openpyxl
import math
import re
import pandas as pd
from transformers import set_seed

## PARÁMETROS
RANDOM_STATE = 42
FILE = 'trolling.xlsx'
OUT_FILE = "rb_db.pkl"
OUT_FILE_F = "rbf_db.pkl"
N = math.inf # Límite de filas a incluir, infinito por ahora
##

set_seed(RANDOM_STATE)


# Función de normalización de texto
def normalize_text(text):
        text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs
        text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios
        text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)|\[removed\]','', text) # He quitado <b><b\> y post eliminados
        return text


data = {
    "txt" : [],
    "label" : [],
}

data_f = {
    "txt" : [],
    "label" : [],
}

wb = openpyxl.load_workbook(FILE)
sheet = wb.active

n = 0
gen = sheet.rows
gen.__next__() # Skip primera fila

for row in gen: # Recordamos, saltando la primera fila
    txt = normalize_text(str(row[0].value))
    vote = [row[1].value,row[2].value,row[3].value]
    rb = vote.count("Trolling")

    filtrar = False
    # Comprobamos que no hayan valores no deseados
    for v in vote:
         if v not in ["Trolling","Normal"]:
            filtrar = True
            break
        
    if len(txt) > 0 and not filtrar: # No añadir filas en blanco o con valores no admitidos
        n+=1
        #if rb in [0,3]:
        data_f['txt'].append(txt)
        data_f['label'].append(math.floor(rb/3 *100)/100)

        data['txt'].append(txt)
        data['label'].append(round(rb/3))
        
        if n >= N: # Cortamos al leer N filas.
            break

# Hace subsample, la clase con menos elementos marcará el máximo de elementos por clase.
def subsample(df):
    uns = [0,1]
    df_classes = []
    c_min = math.inf
    for l in uns:
        c = df[df['label'].round() == l]
        c_min = min(c_min,c.shape[0])
        df_classes.append(c)

    for i,c in enumerate(df_classes):
         df_classes[i] = c.iloc[:c_min]

    df = pd.concat(df_classes)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    return df

# Lo pasamos a dataframe
df = pd.DataFrame(data)
df = subsample(df)
df.to_pickle(OUT_FILE)
print(df.head())

dff = pd.DataFrame(data_f)
dff = subsample(dff)
dff.to_pickle(OUT_FILE_F)
print(dff.head())