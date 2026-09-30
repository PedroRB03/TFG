import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score, matthews_corrcoef, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import configparser
from copy import deepcopy
from scipy.sparse import vstack
from optuna.exceptions import TrialPruned
import time
from pathlib import Path
import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
        "FILE" : 'data/trolling.xlsx',
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
        "LOCAL_VEC_DIR" : "best/lgbm/tfidf", 
        "VFILE" : "best/lgbm/tfidf/tfidf600.pkl", 
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
        "LOCAL_VEC_DIR" : "best/svm/tfidf", 
        "VFILE" : "best/svm/tfidf/tfidf600.pkl", 
        "FILE" : "best/svm/mseed600.pkl", 
    },
    "BERT" : { 
        "OPT_TIME" : 0, 
        "RESULTS" : "results/bert",
        "FUZZY" : True, 
        "EARLY_STOP" : 3,
        "BALANCED_CW" : False,
        "SHOW_LOAD_REPORT" : False, 
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
    with open(fname):
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
def normalize_text(text,light=False):
    text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs.
    text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios.
    if light: # Si light=True no se eliminan [removed]
        text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)','', text) # Quitado &amp;<b><b\>
    else:
        text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)|\[removed\]','', text) # Quitado &amp;<b><b\> y post eliminados.
    return text

# Calcular métricas a partir de resultados de un modelo SVM o LGBM.
def compute_metrics(y_test,y_pred,y_probs):
    return {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'roc_auc_ovr': roc_auc_score(y_test, y_probs),
        'accuracy': accuracy_score(y_test,y_pred),
        'confmat': confusion_matrix(y_test, y_pred)
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
            

        train_ds = pd.read_pickle(file+"/s"+str(seed)+"train.pkl")
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
        print(f"    Tiempo de entrenamiento: {times[i]:.4f}s")
    
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)


    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    print(f"Tiempo de entrenamiento medio: {np.mean(times):.4f}s (+/- {np.std(times):.4f}s)")

# Abre una ventana con una matriz de confusión a partir de las métricas obtenidas de un modelo.
def conf_mat(res,name='confmat'):

    mat = np.array([x[name] for x in res])

    mat_med = np.mean(mat, axis=0)
    mat_std = np.std(mat, axis=0, ddof=1)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=mat_med, display_labels=[0, 1]  # Clases 0 y 1
    )

    disp.plot(cmap=plt.cm.Blues)
    
    for i in range(mat_med.shape[0]):
        for j in range(mat_med.shape[1]):
            disp.text_[i,j].set_text(f"{mat_med[i,j]:.2f} +/- {mat_std[i,j]:.2f}")
    
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    plt.show()

# Usa los mejores parámetros del estudio pasado y evalua el conjunto de datos de test o validación.
# Dejar use_test en True si se quieren usar los conjuntos de datos de test, dejar en False si se quiere usar los de validación.
# También muestra matriz de confusión media y desviación típica.
def test_study(study,model,file,seeds,use_test,save=False,model_path="",NO_TRAIN=False):

    param_grid = study.best_params # Copiamos parámetros desde estudio.

    model.set_params(**param_grid) # Cargamos mejores parámetros.
    results, times = get_results(model,file,seeds,is_test=use_test,save=save,model_path=model_path,NO_TRAIN=NO_TRAIN)
    
    metric_names = ['macro_f1', 'balanced_accuracy', 'matthews_corrcoef', 'roc_auc_ovr','accuracy']

    printtest(seeds,results,times,metric_names)
    conf_mat(results)
  