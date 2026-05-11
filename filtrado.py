import openpyxl
import math
import re
import pandas as pd
from sklearn.model_selection import train_test_split

## PARÁMETROS
SEEDS = [600,601,602,603,604]
TEST_PERCENT = 15
EVAL_PERCENT = 15
FILE = 'trolling.xlsx'
OUT_FILE = "pkls/rb_db"
OUT_FILE_F = "pkls/rbf_db"
N = math.inf # Límite de filas a incluir, infinito por ahora
VERBOSE = False # Mostrar los primeros elementos de cada archivo
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

    df = pd.concat(df_classes).reset_index(drop=True)
    return df

# Separa datos en tres subconjuntos: entrenamiento, evaluación y test.
def make_df(df,test_size,eval_size,seed=1):
    #df = pd.DataFrame(data)
    #df = df.drop_duplicates(subset=['txt'])
    #df = subsample(df)
    df_train, df_testeval = train_test_split(df, test_size=test_size/100+eval_size/100, stratify=df['label'], random_state=seed)
    df_eval, df_test = train_test_split(df_testeval, test_size=test_size/(test_size+eval_size), stratify=df_testeval['label'], random_state=seed)
    return df_train, df_eval, df_test

# Creamos dataframes sin barajar 
df = pd.DataFrame(data)
df = df.drop_duplicates(subset=['txt']) # quitamos duplicados
df = subsample(df)
dff = pd.DataFrame(data_f)
dff = dff.drop_duplicates(subset=['txt']) # quitamos duplicados
dff = subsample(dff)

# Por cada semilla creamos archivos de train, eval y test para crisp y fuzzy.
for seed in SEEDS:
    if VERBOSE:
        print("SEED: "+str(seed))

        print("CRISP")
    df_tr, df_ev,df_tst = make_df(df,TEST_PERCENT,EVAL_PERCENT,seed)
    if VERBOSE:
        print(f"TRAIN \n{df_tr.head()}")
        print(f"EVAL \n{df_ev.head()}")
        print(f"TEST \n{df_tst.head()}")
    df_tr.to_pickle(OUT_FILE+str(seed)+"train.pkl")
    df_ev.to_pickle(OUT_FILE+str(seed)+"eval.pkl")
    df_tst.to_pickle(OUT_FILE+str(seed)+"test.pkl")

    #if len(set(df_tr["txt"]).intersection(set(df_ev["txt"]))) > 0 or len(set(df_tst["txt"]).intersection(set(df_ev["txt"]))) > 0:
    #    print("Hay elementos comunes en los datasets...")
    #    break

    if VERBOSE:
        print("FUZZY")
    dff_tr, dff_ev,dff_tst = make_df(dff,TEST_PERCENT,EVAL_PERCENT,seed)
    if VERBOSE:
        print(f"\nTRAIN \n{dff_tr.head()}")
        print(f"\nEVAL \n{dff_ev.head()}")
        print(f"\nTEST \n{dff_tst.head()}")
    dff_tr.to_pickle(OUT_FILE_F+str(seed)+"train.pkl")
    dff_ev.to_pickle(OUT_FILE_F+str(seed)+"eval.pkl")
    dff_tst.to_pickle(OUT_FILE_F+str(seed)+"test.pkl")

    #if len(set(dff_tr["txt"]).intersection(set(dff_ev["txt"]))) > 0 or len(set(dff_tst["txt"]).intersection(set(dff_ev["txt"]))) > 0:
    #    print("Hay elementos comunes en los datasets...")
    #    break


