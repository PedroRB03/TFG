import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import numpy as np
import re
import configparser
from copy import deepcopy
from scipy.sparse import vstack
from math import inf

from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


PARAM_DEFAULTS = {
    "COMMON" : { 
        "SEEDS" : [600,601,602,603,604],   
        "FILE" : "rb_db", 
        "FILE_F" : "rbf_db", 
        "VFILE" : "data/pkls/crisp/rb_db", 
    },
    "DATAGEN" : {
        "TEST_PERCENT" : 15,
        "EVAL_PERCENT" : 15,
        "FILE" : 'trolling.xlsx',
        "N" : inf, 
        "VERBOSE" : False, 
        "CHECK_COL" : False, 
        "BALANCING" : "None", 
        "NGRAM_MIN" : 1, 
        "NGRAM_MAX" : 3,
    },
    "LGBM" : { 
        "OPT_TIME" : 60, 
        "OPT_STUDY" : "lgbm-study", 
        "OPT_DB" : "sqlite:///lgbm-study.db", 
        "MODEL_PATH" : "", 
        "USE_TEST" : False, 
        "SAVE_MODEL" : False, 
    },
    "SVM" : {
        "OPT_TIME" : 60, 
        "OPT_STUDY" : "svm-study", 
        "OPT_DB" : "sqlite:///svm-study.db",
        "MODEL_PATH" : "", 
        "USE_TEST" : False,
        "SAVE_MODEL" : False,
    },
    "BERT" : { 
        "OPT_TIME" : 60, 
        "OPT_STUDY" : "bert-study", 
        "OPT_DB" : "sqlite:///bert-study.db", 
        "RESULTS" : "",
        "FUZZY" : False, 
        "USE_TEST" : False, 
        "EARLY_STOP" : 3,
        "SAVE_MODEL" : False, 
    },
}

def get_params(fname="params.ini"):
    config = configparser.ConfigParser()

    res = {}
    config.read(fname)
        
    for csec in ["COMMON","LGBM","SVM","BERT","DATAGEN"]:
        if csec not in config.keys():
            res[csec] = deepcopy(PARAM_DEFAULTS[csec])
        else:
            res[csec] = {}
            for param in PARAM_DEFAULTS[csec].keys():
                if param not in config[csec].keys():
                    res[csec][param] = PARAM_DEFAULTS[csec][param]
                else:
                    match PARAM_DEFAULTS[csec][param]:
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
    if res["DATAGEN"]["N"] == 0:
        res["DATAGEN"]["N"] = inf
    return res



# Función de normalización de texto
def normalize_text(text):
        text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs
        text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios
        text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)|\[removed\]','', text) # He quitado <b><b\> y post eliminados
        return text


# balancing puede ser None (Ninguna ténica de balanceo), RUS (RandomUnderSampler), ENN (EditedNearestNeighbours) o SMOTE
def balance(df,balancing,seed):
    bal = None
    match balancing:
        case "RUS":
            bal = RandomUnderSampler(random_state=seed)
        case "ENN":
            bal = EditedNearestNeighbours()
        case "ROS":
            bal = RandomOverSampler(random_state=seed)

    if bal:
        y = df['label']
        y = (y*100).round().astype(int)
        x = vstack(df["vec"].to_list())
        _, _ = bal.fit_resample(x,y)

        df = df.iloc[bal.sample_indices_].reset_index(drop=True)
    
    #x,y = shuffle(x,y,random_state=seed)
    #return x,y
    return df.sample(frac=1,random_state=seed).reset_index(drop=True)

# Calcular métricas a partir de resultados de un modelo
def compute_metrics(y_test,y_pred,y_probs):
    return {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'roc_auc_ovr': roc_auc_score(y_test, y_probs)
    }

# Obtener métricas a partir de un modelo y distintas semillas
def get_results(model,file,seeds,is_test=False):
    results = []
    models = {}

    for seed in seeds:
        train_ds = pd.read_pickle(file+str(seed)+"train.pkl")
        if is_test:
            test_ds = pd.read_pickle(file+str(seed)+"test.pkl")
        else:
            test_ds = pd.read_pickle(file+str(seed)+"eval.pkl")
        
        X_trn = vstack(train_ds["vec"].to_list())
        X_tst = vstack(test_ds["vec"].to_list())
        Y_trn = train_ds["label"]
        Y_tst = test_ds["label"]
        #if balancing is not None:
        #    X_trn, Y_trn = balance(X_trn,Y_trn,balancing,seed)

        model.fit(X_trn,Y_trn)
        y_pred = model.predict(X_tst)
        y_probs = model.predict_proba(X_tst)[:,1]
        models[seed] = model
        results.append(compute_metrics(Y_tst,y_pred,y_probs))

    return results, models

# Crea un vectorizador con el rango de ngram dado
def make_vectorizer(ngram_range):
    return TfidfVectorizer( # Vectorizador
        ngram_range=ngram_range,    
        stop_words="english",
        #lowercase=True
    )

# Busca los mejores parámetros para el estudio y pipeline pasado
# Necesita una función objective que acepte el pipeline como primer parámetro
def optimize(study,objective,timeout):
    study.optimize(objective, timeout=timeout)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")

# Usa los mejores parámetros del estudio pasado y evalua el conjunto de datos de test o evaluación
# Dejar use_test en True si se quieren usar los conjuntos de datos de test, dejar en False si se quiere usar los de evaluación
def test_study(study,model,file,seeds,use_test):
    #max_n = study.best_params['pre__tfidf__ngram_range_max'] 
    #min_n = study.best_params['pre__tfidf__ngram_range_min']

    param_grid = study.best_params # Copiamos parámetros desde estudio
    #param_grid.pop('pre__tfidf__ngram_range_max') # max y min no existen realmente en tfidf, los quitamos
    #param_grid.pop('pre__tfidf__ngram_range_min')

    #tfidf = make_vectorizer((min_n,max_n))
    model.set_params(**param_grid) # Cargamos mejores parámetros
    results, models = get_results(model,file,seeds,is_test=use_test)
    
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)

    metric_names = ['macro_f1', 'balanced_accuracy', 'matthews_corrcoef', 'roc_auc_ovr']
    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    return models