import openpyxl
import math
import re
import pandas as pd
from sklearn.model_selection import train_test_split

## PARÁMETROS
SEEDS = [600,601,602,603,604]
TEST_PERCENT = 15
EVAL_PERCENT = 15
FILE = 'data/trolling.xlsx'
OUT_FILE = "data/pkls/crisp/rb_db"
OUT_FILE_F = "data/pkls/fuzzy/rbf_db"
N = math.inf # Límite de filas a incluir, infinito por ahora
VERBOSE = False # Mostrar los primeros elementos de cada archivo
CHECK_COL = False # Comprobar si hay colisiones entre los datos de entrenamiento, test y evaluación
##



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
gen.__next__() # Saltar primera fila

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
        data_f['label'].append(round(rb/3 *100)/100)

        data['txt'].append(txt)
        data['label'].append(round(rb/3))
        
        if n >= N: # Cortamos al leer N filas.
            break



# Separa datos en tres subconjuntos: entrenamiento, evaluación y test.
def make_df(df,test_size,eval_size,seed=1):
    df_train, df_testeval = train_test_split(df, test_size=test_size/100+eval_size/100, stratify=df['label'], random_state=seed)
    df_eval, df_test = train_test_split(df_testeval, test_size=test_size/(test_size+eval_size), stratify=df_testeval['label'], random_state=seed)
    
    return df_train.reset_index(drop=True), df_eval.reset_index(drop=True), df_test.reset_index(drop=True)

# no usada, quitar en versión final
def remNcheckDupes(df):
    print(f"Duplicados CRISP normalizado: {len(df['txt'])-len(df['txt'].drop_duplicates())}")
    df = df.drop_duplicates(subset=['txt']) # quitamos duplicados
    print(f"Duplicados CRISP barajado: {len(df['txt'])-len(df['txt'].drop_duplicates())}")

def distnel(l1,l2):
    return len(set(l1).intersection(set(l2))) > 0 
# Creamos dataframes
df = pd.DataFrame(data)
df = df.drop_duplicates(subset=['txt']) # quitamos duplicados


dff = pd.DataFrame(data_f)
dff = dff.drop_duplicates(subset=['txt']) # quitamos duplicados

dfs = [df, dff]
filepaths = [OUT_FILE,OUT_FILE_F]
titles = ["CRISP","FUZZY"]

# Por cada semilla creamos archivos de train, eval y test para crisp y fuzzy.
for seed in SEEDS:
    print("SEED: "+str(seed))

    for i in range(0,len(dfs)):
        df_i = dfs[i]
        f_path = filepaths[i]
        title = titles[i]

        print(title)
        df_tr, df_ev,df_tst = make_df(df_i,TEST_PERCENT,EVAL_PERCENT,seed)
        if VERBOSE:
            print(f"TRAIN \n{df_tr.head()}")
            print(f"EVAL \n{df_ev.head()}")
            print(f"TEST \n{df_tst.head()}")
        df_tr.to_pickle(f_path+str(seed)+"train.pkl")
        df_ev.to_pickle(f_path+str(seed)+"eval.pkl")
        df_tst.to_pickle(f_path+str(seed)+"test.pkl")

        if CHECK_COL and (distnel(df_tr["txt"],df_ev["txt"]) or distnel(df_tst["txt"],df_ev["txt"])):
            print("Hay elementos comunes en los datasets...")
            print("Dataset: "+title+" en semilla: "+str(seed))
            break


print("Ficheros generados.")