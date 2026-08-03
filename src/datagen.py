import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split
from common import get_params,normalize_text,make_vectorizer, balance
import sys
import joblib
from pathlib import Path




if __name__ == "__main__":
            
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")

    SEEDS = params["COMMON"]["SEEDS"]
    TEST_PERCENT = params["DATAGEN"]["TEST_PERCENT"]
    EVAL_PERCENT = params["DATAGEN"]["EVAL_PERCENT"]
    FUZZY_BAL_CRISP = params["COMMON"]["FUZZY_BAL_CRISP"]
    FILE = params["DATAGEN"]["FILE"]
    VFILE = params["COMMON"]["VFILE"]
    OUT_FILE = params["COMMON"]["FILE"]
    OUT_FILE_F = params["COMMON"]["FILE_F"]
    N = params["DATAGEN"]["N"]
    VERBOSE = params["DATAGEN"]["VERBOSE"]
    CHECK_COL = params["DATAGEN"]["CHECK_COL"]
    BALANCING = params["DATAGEN"]["BALANCING"]
    NGRAM_MIN = params["DATAGEN"]["NGRAM_MIN"]
    NGRAM_MAX = params["DATAGEN"]["NGRAM_MAX"]
    ##

    data = { # datos crisp
        "txt" : [],
        "label" : [],
    }

    data_f = { # datos fuzzy
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
    def data_separate(df,test_size,eval_size,seed=1):
        if (test_size+eval_size) >= 100: # Comprobamos validez de los porcentajes
            raise Exception("test_size y eval_size no pueden sumar 100 o valores superiores.")
        df_train, df_testeval = train_test_split(df, test_size=test_size/100+eval_size/100, stratify=df['label'], random_state=seed)
        df_eval, df_test = train_test_split(df_testeval, test_size=test_size/(test_size+eval_size), stratify=df_testeval['label'], random_state=seed)
        # Reseteamos índices
        df_train=df_train.reset_index(drop=True)
        df_eval=df_eval.reset_index(drop=True)
        df_test=df_test.reset_index(drop=True)

        return df_train, df_eval, df_test

    def distnel(l1,l2):
        return len(set(l1).intersection(set(l2))) > 0 

    # Creamos dataframes
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['txt']) # Quitamos duplicados


    dff = pd.DataFrame(data_f)
    dff = dff.drop_duplicates(subset=['txt']) # Quitamos duplicados


    
    dfs = [df, dff] 
    tfidfs = ["/tfidf_crisp","/tfidf_fuzzy"]
    filepaths = [OUT_FILE+"/rb_db",OUT_FILE_F+"/rbf_db"]
    titles = ["CRISP","FUZZY"]

    # Por cada semilla creamos archivos de train, eval y test para crisp y fuzzy.
    for seed in SEEDS:
        print("SEED: "+str(seed))

        for i in range(0,len(dfs)): # i = 0 cuando CRISP, i = 1 cuando FUZZY
            df_i = dfs[i]
            f_path = filepaths[i]
            tfidfp = tfidfs[i]
            title = titles[i]

            print(title)
            df_tr, df_ev,df_tst = data_separate(df_i,TEST_PERCENT,EVAL_PERCENT,seed)
            tfidf = make_vectorizer((NGRAM_MIN,NGRAM_MAX))
            vec_tr = tfidf.fit_transform(df_tr["txt"])
            vec_ev = tfidf.transform(df_ev["txt"])
            vec_tst = tfidf.transform(df_tst["txt"])

            Path(VFILE).mkdir(parents=True, exist_ok=True)
            joblib.dump(tfidf, VFILE+tfidfp+str(seed)+".pkl")
            
            df_tr['vec'] = [vec_tr.getrow(i) for i in range(vec_tr.shape[0])]
            df_ev['vec'] = [vec_ev.getrow(i) for i in range(vec_ev.shape[0])]
            df_tst['vec'] = [vec_tst.getrow(i) for i in range(vec_tst.shape[0])]

            if BALANCING is not None and BALANCING != "None" and BALANCING != "": # Balanceamos si BALANCING no es None
                df_tr = balance(df_tr,BALANCING,seed,FUZZY_BAL_CRISP=FUZZY_BAL_CRISP)
                
            if VERBOSE:
                print(f"TRAIN \n{df_tr.tail()}")
                print(f"EVAL \n{df_ev.tail()}")
                print(f"TEST \n{df_tst.tail()}")
                
            df_tr.to_pickle(f_path+str(seed)+"train.pkl")
            df_ev.to_pickle(f_path+str(seed)+"eval.pkl")
            df_tst.to_pickle(f_path+str(seed)+"test.pkl")

            if CHECK_COL and (distnel(df_tr["txt"],df_ev["txt"]) or distnel(df_tst["txt"],df_ev["txt"])):
                print("Hay elementos comunes en los datasets...")
                print("Dataset: "+title+" en semilla: "+str(seed))
                break


    print("Ficheros generados.")