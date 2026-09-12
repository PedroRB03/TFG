
import pandas as pd
from sklearn.model_selection import train_test_split
from common import get_params,normalize_text,make_vectorizer, balance
import sys
import joblib
from pathlib import Path


# Separa datos en tres subconjuntos: entrenamiento, validación y test.
def _data_separate(df,test_size,eval_size,seed=1):
    if (test_size+eval_size) >= 100: # Comprobamos validez de los porcentajes.
        raise Exception("test_size y eval_size no pueden sumar 100 o valores superiores.")
    
    df_train, df_testeval = train_test_split(df, test_size=test_size/100+eval_size/100, stratify=df['label'], random_state=seed)

    df_eval, df_test = train_test_split(df_testeval, test_size=test_size/(test_size+eval_size), stratify=df_testeval['label'], random_state=seed)
    # Reseteamos índices.
    df_train=df_train.reset_index(drop=True)
    df_eval=df_eval.reset_index(drop=True)
    df_test=df_test.reset_index(drop=True)

    return df_train, df_eval, df_test


# Obtiene y prepara el DataFrame pre-procesado a partir de una ruta al dataset.
def get_dfs(FILE):
    # Creamos dataframes.
    df = pd.read_excel(FILE)
    # Renombramos columnas.
    df.columns = ['txt','a1','a2','a3','label']

    # Eliminamos filas con valores no deseados.
    allowed = ["Normal","Trolling"]
    df = df[df['a1'].isin(allowed) & df['a2'].isin(allowed) & df['a3'].isin(allowed)]
    
    # Normalizamos texto.
    df['txt'] = df['txt'].apply(lambda x: normalize_text(str(x)))

    # Reemplazamos "Normal" y "Trolling" por 0 y 1 respectivamente.
    rep_list = ['a1','a2','a3','label']
    mapping = {'Normal':0, 'Trolling': 1}
    df[rep_list] = df[rep_list].replace(mapping)

    # Obtenemos valores fuzzy.

    df['labelf'] = df[['a1','a2','a3']].mean(axis=1)
    
    # Eliminamos columnas no útiles.
    df = df[['txt','label','labelf']]
    df.columns = ['txt','label','labelf']

    # Quitamos duplicados.
    df = df.drop_duplicates(subset=['txt'],keep=False,ignore_index=True) 
    return df

if __name__ == "__main__":
            
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")

    SEEDS = params["COMMON"]["SEEDS"]
    TEST_PERCENT = params["DATAGEN"]["TEST_PERCENT"]
    EVAL_PERCENT = params["DATAGEN"]["EVAL_PERCENT"]
    FILE = params["DATAGEN"]["FILE"]
    VFILE = params["COMMON"]["VFILE"]
    OUT_FILE = params["COMMON"]["FILE"]
    BALANCING = params["DATAGEN"]["BALANCING"]
    NGRAM_MIN = params["DATAGEN"]["NGRAM_MIN"]
    NGRAM_MAX = params["DATAGEN"]["NGRAM_MAX"]
    ##

    df = get_dfs(FILE)


    filepaths = [OUT_FILE+"/rb_db",OUT_FILE+"/rbf_db"] # Ruta y prefijo de los conjuntos de datos a guardar.
    

    # Por cada semilla creamos archivos de train, eval y test para crisp y fuzzy.
    for seed in SEEDS:
        print("SEED: "+str(seed))

        # Separamos datos
        df_tr, df_ev,df_tst = _data_separate(df,TEST_PERCENT,EVAL_PERCENT,seed)
        # Creamos y entrenamos vectorizadores
        tfidf = make_vectorizer((NGRAM_MIN,NGRAM_MAX))
        vec_tr = tfidf.fit_transform(df_tr["txt"])
        vec_ev = tfidf.transform(df_ev["txt"])
        vec_tst = tfidf.transform(df_tst["txt"])
        

        # Creamos la columna vec con las entradas vectorizadas
        df_tr['vec'] = [vec_tr[i,:] for i in range(vec_tr.shape[0])]
        df_ev['vec'] = [vec_ev[i,:] for i in range(vec_ev.shape[0])]
        df_tst['vec'] = [vec_tst[i,:] for i in range(vec_tst.shape[0])]

        # Separamos entrenamiento crisp y fuzzy
        dff_tr = df_tr.copy()
        
        if BALANCING is not None: # Balanceamos si BALANCING no es None.
            df_tr = balance(df_tr,BALANCING,seed,FUZZY_BAL_CRISP=False)
            if BALANCING == "RUS": # Solo RUS está disponible para fuzzy
                dff_tr = balance(dff_tr,BALANCING,seed,FUZZY_BAL_CRISP=True) # Se consideran etiquetas crisp para el balanceo
            

        # Creamos rutas si no existen
        Path(VFILE).mkdir(parents=True, exist_ok=True)
        Path(OUT_FILE).mkdir(parents=True, exist_ok=True)

        # Guardamos vectorizador y conjuntos de entrenamiento, validación y test.
        joblib.dump(tfidf, VFILE+"/tfidf"+str(seed)+".pkl")
        df_tr.to_pickle(OUT_FILE+"/cs"+str(seed)+"train.pkl")
        dff_tr.to_pickle(OUT_FILE+"/fs"+str(seed)+"train.pkl")
        df_ev.to_pickle(OUT_FILE+"/s"+str(seed)+"eval.pkl")
        df_tst.to_pickle(OUT_FILE+"/s"+str(seed)+"test.pkl")
        
    print("Ficheros generados.")