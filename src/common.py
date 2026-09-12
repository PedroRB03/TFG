import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score, matthews_corrcoef, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from datetime import timedelta
import configparser
from copy import deepcopy
from scipy.sparse import vstack
from math import inf
from sklearn.utils.class_weight import compute_class_weight
from optuna.exceptions import TrialPruned
import time
from pathlib import Path
import joblib
from sklearn.svm import SVC

from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Parámetros por defecto.
PARAM_DEFAULTS = {
    "COMMON" : { 
        "SEEDS" : [600,601,602,603,604],   
        "VFILE" : "results/tfidf", 
        "FILE" : "data/pkls", 
    },
    "DATAGEN" : {
        "TEST_PERCENT" : 15,
        "EVAL_PERCENT" : 15,
        "FILE" : 'trolling.xlsx',
        "STUDY_OUT" : "results/analytics",
        "BALANCING" : "None", 
        "NGRAM_MIN" : 1, 
        "NGRAM_MAX" : 3,
    },
    "LGBM" : { 
        "OPT_TIME" : 0, 
        "NO_TRAIN" : False,
        "OPT_STUDY" : "lgbm-study", 
        "OPT_DB" : "sqlite:///results/lgbm-study.db", 
        "MODEL_PATH" : "results/lgbm", 
        "USE_TEST" : False, 
        "SAVE_MODEL" : False, 
        "LOCAL_MODEL_DIR" : "best/lgbm", 
        "VFILE" : "best/tfidf/tfidf600.pkl", 
        "FILE" : "best/lgbm/mseed600.pkl", 
    },
    "SVM" : {
        "OPT_TIME" : 0, 
        "KERNEL" : "linear", 
        "NO_TRAIN" : False,
        "OPT_STUDY" : "svm-study", 
        "OPT_DB" : "sqlite:///results/svm-study.db",
        "MODEL_PATH" : "results/svm", 
        "USE_TEST" : False,
        "SAVE_MODEL" : False,
        "LOCAL_MODEL_DIR" : "best/svm", 
        "VFILE" : "best/tfidf/tfidf600.pkl", 
        "FILE" : "best/svm/mseed600.pkl", 
    },
    "BERT" : { 
        "OPT_TIME" : 0, 
        "RESULTS" : "results/bert",
        "FUZZY" : True, 
        "EARLY_STOP" : 3,
        "BALANCED_CW" : False,
        "SHOW_LOAD_REPORT" : True, 
        "NO_TRAIN" : False,
        "OPT_STUDY" : "bert-study", 
        "OPT_DB" : "sqlite:///results/bert-study.db", 
        "USE_TEST" : False, 
        "SAVE_MODEL" : False, 
        "LOCAL_MODEL_DIR" : "best/bert", 
        "LOCAL_MODEL_NAME" : "best/bert/bert_model600", 
    },
}

# Carga los parámetros los devuelve en un diccionario.
def get_params(fname="params.ini"):
    config = configparser.ConfigParser()

    res = {}
    config.read(fname)
        
    for csec in ["COMMON","LGBM","SVM","BERT","DATAGEN"]: # Cada sección.
        if csec not in config.keys():
            res[csec] = deepcopy(PARAM_DEFAULTS[csec]) # En caso de no encontrarse se pone el valor por defecto.
        else:
            res[csec] = {}
            for param in PARAM_DEFAULTS[csec].keys(): # Cada parámetro por sección.
                if param not in config[csec].keys():
                    res[csec][param] = PARAM_DEFAULTS[csec][param] # En caso de no encontrarse se pone el valor por defecto.
                else:
                    match PARAM_DEFAULTS[csec][param]: # Typecasting y carga de cada parámetro.
                        case float():
                            res[csec][param] = float(config[csec][param])
                        case bool():
                            res[csec][param] = (config[csec][param] == "True" or config[csec][param] == "true")
                        case int():
                            res[csec][param] = int(config[csec][param])
                        case list():
                            arr = config[csec][param].split(";")
                            res[csec][param] = []
                            for i in arr:
                                res[csec][param].append(int(i))
                        case _:
                            res[csec][param] = config[csec][param]
   
    return res



# Función de normalización de texto.
def normalize_text(text):
    text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs.
    text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios.
    text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)|\[removed\]','', text) # Quitado &amp;<b><b\> y post eliminados.
    return text

# Esta función devuelve el índice del elemento más cercano al valor dado de una lista.
def _closest(v,l):
    i = 0
    fi = len(l)
    d = inf
    for e in l:
        di = abs(v-e)
        if di < d:
            fi = i
            d = di
        i+=1
    return fi

# Devuelve los pesos de clase de la lista dada. Si FUZZY_BAL_CRISP=True, se redondean las etiquetas para el cálculo.
def get_class_weights(arr,FUZZY_BAL_CRISP=True):

    if FUZZY_BAL_CRISP: 
        farr = [int(x >= 0.5) for x in arr]
    else:
        farr = arr

    classes = np.unique(farr)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=farr
    )

    if FUZZY_BAL_CRISP:
        return [class_weights[_closest(x,classes)] for x in np.unique(arr)]

    return class_weights

# Aplica sobre df la técnica de balanceo especificada por balancing y si FUZZY_BAL_CRISP=True se emplean clases crisp aunque el dataset sea fuzzy
# balancing puede ser None (Ninguna ténica de balanceo), RUS (RandomUnderSampler), ENN (EditedNearestNeighbours) o SMOTE.
def balance(df,balancing,seed,FUZZY_BAL_CRISP=False):
    bal = None
    match balancing: # Creamos clase correspondiente.
        case "RUS":
            bal = RandomUnderSampler(random_state=seed)
        case "ENN":
            bal = EditedNearestNeighbours()
        case "SMOTE":
            bal = SMOTE(random_state=seed)

    if balancing != "RUS" and FUZZY_BAL_CRISP:
        raise Exception("FUZZY_BAL_CRISP solo puede ser verdadero con RUS")
    
    if bal:
        is_float = df['label'].dtype != 'int64'
        if FUZZY_BAL_CRISP:
            y = (df['label']).round().astype(int)
        else:
            y = np.floor(df['label']*3).astype(int)

        x = vstack(df["vec"])
        x_new, y_new = bal.fit_resample(x,y)

        n_og = len(df)
        n_new = len(y_new)
        if balancing == "SMOTE": # En caso de SMOTE, adjuntamos casos sintéticos al dataset principal.
            if n_new > 0:
                new_rows = {'txt':[],'label':[],'vec':[]}
                for i in range(n_og,n_new):
                    y_i = y_new[i]
                    if is_float:
                        y_i = float(y_i)/3
                    else:
                        y_i = int(y_i)/3
                    vec_i = x_new[i]
                    txt_rng = "" # Se deja un texto de relleno por compatibilidad, la columna txt NO se debe utilizar cuando se emplea SMOTE
                    new_rows['txt'].append(txt_rng)
                    new_rows['label'].append(y_i)
                    new_rows['vec'].append(vec_i)

                df = pd.concat([df,pd.DataFrame(new_rows)], ignore_index=True)
        else:
            df = df.iloc[bal.sample_indices_]
    
    return df.sample(frac=1,random_state=seed).reset_index(drop=True) # Barajamos una última vez.

# Calcular métricas a partir de resultados de un modelo SVM o LGBM.
def compute_metrics(y_test,y_pred,y_probs):
    return {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'roc_auc_ovr': roc_auc_score(y_test, y_probs),
        'accuracy': accuracy_score(y_test,y_pred)
    }

# Obtener métricas a partir de un modelo y distintas semillas. También devuelve el modelo entrenado de cada semilla.
def get_results(model,file,seeds,is_test=False,trial=None,save=False,model_path="",NO_TRAIN=False):
    results = []

    fit_t = []

    for step, seed in enumerate(seeds):
        if trial is not None:
            if trial.should_prune():
                raise TrialPruned(f"Podada semilla {seed} por decisión del pruner.")
            if should_prune(trial,results,len(seeds)):
                raise TrialPruned(f"Podada semilla {seed} por media optimista inferior al mejor estudio.")
            

        train_ds = pd.read_pickle(file+"/cs"+str(seed)+"train.pkl")
        if is_test: # Utilizamos conjunto de validación o test dependiendo del parámetros is_test.
            test_ds = pd.read_pickle(file+"/s"+str(seed)+"test.pkl")
        else:
            test_ds = pd.read_pickle(file+"/s"+str(seed)+"eval.pkl")

        
        
        X_trn = vstack(train_ds["vec"]) # Formamos matriz dispersa.
        Y_trn = train_ds["label"].astype("category")

        X_tst = vstack(test_ds["vec"]) # Formamos matriz dispersa.
        Y_tst = test_ds["label"].astype("category")

        if NO_TRAIN:
            model = joblib.load(model_path+"/mseed"+str(seed)+".pkl")
            fit_t.append(0)
        else:
            model.set_params(random_state=seed)

            st = time.time()
            model.fit(X_trn,Y_trn)
            fit_t.append(time.time()-st)

        y_pred = model.predict(X_tst)
        if type(model) == SVC:
            y_probs = model.decision_function(X_tst) # distancia al hiperplano para SVC en lugar de probabilidades
        else:
            y_probs = model.predict_proba(X_tst)[:,1]

        if save and not NO_TRAIN:
            Path(model_path).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path+"/mseed"+str(seed)+".pkl")
            print(f"Guardado modelo de semilla: {seed}")
       
        r = compute_metrics(Y_tst,y_pred,y_probs)
        if trial is not None:
            trial.report(r['macro_f1'],step=step)
        results.append(r)


    return results, fit_t

# Crea un vectorizador con el rango de ngram dado.
def make_vectorizer(ngram_range):
    return TfidfVectorizer( # Vectorizador.
        ngram_range=ngram_range,    
        stop_words="english",
        #lowercase=True
    )

# Obtiene una media optimista en mitad de un trial de Optuna y decide si se debería podar el intento.
def should_prune(trial,results,n,eval_name='macro_f1'):
    try:
        bf1 = trial.study.best_value
    except ValueError:
        bf1 = -float('inf')
    i = 0
    med = 0
    for r in results:
        med+=r[eval_name]
        i+=1
    for j in range(i,n):
        med+=1
    return med/n < bf1

# Busca los mejores parámetros para el estudio y pipeline pasado.
# Necesita una función objective que acepte el pipeline como primer parámetro.
def optimize(study,objective,timeout):
    study.optimize(objective, timeout=timeout)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")

# Dada una semilla, los resultados, tiempos y nombres de métricas, las muestra por pantalla.
def printtest(seeds,results,times,metric_names):

    print("\n" + "="*30)
    print("RESULTADOS POR SEMILLA")
    print("="*30)

    for i,seed in enumerate(seeds):
        print(f"Semilla: {seed}")
        for m in metric_names:
            values = [r[m] for r in results]
            print(f"    {m}: {values[i]:.4f}")
        print(f"    Tiempo de entrenamiento: {timedelta(milliseconds=int(times[i]*1000))}")
    
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)


    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    print(f"Tiempo de entrenamiento medio: {timedelta(milliseconds=int(np.mean(times)*1000))}s (+/- {np.std(times):.4f}s)")

# Usa los mejores parámetros del estudio pasado y evalua el conjunto de datos de test o validación.
# Dejar use_test en True si se quieren usar los conjuntos de datos de test, dejar en False si se quiere usar los de validación.
def test_study(study,model,file,seeds,use_test,save=False,model_path="",NO_TRAIN=False):

    param_grid = study.best_params # Copiamos parámetros desde estudio.

    model.set_params(**param_grid) # Cargamos mejores parámetros.
    results, times = get_results(model,file,seeds,is_test=use_test,save=save,model_path=model_path,NO_TRAIN=NO_TRAIN)
    
    metric_names = ['macro_f1', 'balanced_accuracy', 'matthews_corrcoef', 'roc_auc_ovr','accuracy']

    printtest(seeds,results,times,metric_names)
    