import openpyxl
import csv
import math
import re
import pandas as pd
from transformers import set_seed

## PARÁMETROS
RANDOM_STATE = 42
FILE = 'trolling.xlsx'
OUT_FILE = "rb_db.pkl"
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

#with open('ragebait.csv','w',encoding='utf-8') as f:

    #writer = csv.writer(f,quoting=csv.QUOTE_ALL) # Lo vamos a exportar en csv también
    #writer.writerow(['txt','label'])

wb = openpyxl.load_workbook(FILE)
sheet = wb.active

n = 0
gen = sheet.rows
gen.__next__() # Skip primera fila

N = math.inf # Límite de filas a leer, infinito por ahora
for row in gen: # Recordamos, saltando la primera fila
    n+=1
    txt = normalize_text(str(row[0].value))
    rb = [row[1].value,row[2].value,row[3].value].count("Trolling")

    #if len(txt) > 0: 
    #    writer.writerow([txt, rb])

    if len(txt) > 0: # No añadir filas en blanco
        if rb in [0,3]:
            v = 0 if rb == 0 else 1
            data['txt'].append(txt)
            data['label'].append(v)
        if n >= N: # Cortamos al leer N filas.
            break

# Lo pasamos a dataframe
df = pd.DataFrame(data)

# Hacemos subsampling
df_clase_0 = df[df['label'] == 0]
df_clase_1 = df[df['label'] == 1]

Min = min(df_clase_0.shape[0],df_clase_1.shape[0]) # min del nº de elementos entre las dos clases

df_clase_0 = df_clase_0.iloc[:Min] # Redimensionamiento
df_clase_1 = df_clase_1.iloc[:Min]

df_equilibrado = pd.concat([df_clase_0, df_clase_1]) # Juntamos clases
df_equilibrado = df_equilibrado.sample(frac=1).reset_index(drop=True) # shuffle


df_equilibrado.to_pickle(OUT_FILE)
print(df_equilibrado.head())
