import pandas as pd
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score, matthews_corrcoef
from sklearn import svm
import numpy as np


## PARÁMETROS
OPT_TIME = 0 # Dejar en 0 si no se quieren buscar hiperparámetros
OPT_STUDY = "svm-study" # nombre del estudio de optuna
OPT_DB = "sqlite:///svm-study.db" # nombre de la base de datos con estudio
SEEDS = [600,601,602,603,604] # semillas que se usarán
FILE = "pkls/rb_db" # prefijo de archivos a usar
##

## PIPELINE
tfidf_p = TfidfVectorizer( # Vectorizador
    #ngram_range=(1,3),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'txt') # solo vamos a vectorizar
    ],
    remainder='drop'  
)

model = svm.NuSVC( # Hiperparámetros
    #nu=0.22163740307384033,
    decision_function_shape="ovr",
    #kernel='linear',
    #gamma=0.025777463245085116,
    probability=True
)

pipeline = Pipeline([
    ('prepro', preprocessor), # vectorización
    ('clf', model) # clasificador
]) 
##

# Función para calcular métricas
def compute_metrics(y_test,y_pred):
    return {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'roc_auc_ovr': roc_auc_score(y_test, y_pred)
    }

# Función para obtener métricas de las semillas
def get_results(file,seeds):
    results = []

    for seed in seeds:
        train_ds = pd.read_pickle(file+str(seed)+"train.pkl")
        test_ds = pd.read_pickle(file+str(seed)+"test.pkl")
        pipeline.fit(train_ds.drop(columns="label"),train_ds["label"])
        y_pred = pipeline.predict(test_ds.drop(columns="label"))
        
        results.append(compute_metrics(test_ds["label"],y_pred))

    return results

# Función para buscar hiperparámetros
def objective(trial):
    max_n = trial.suggest_int('prepro__post_tfidf__ngram_range_max', 2, 5)
    min_n = trial.suggest_int('prepro__post_tfidf__ngram_range_min', 1, 2)
    kernel = trial.suggest_categorical('clf__kernel', ['linear', 'rbf', 'poly'])

    param_grid = {
        'prepro__post_tfidf__ngram_range': (min_n, max_n),
        'clf__nu': trial.suggest_float('clf__nu', 0.1, 0.5) ,
        'clf__gamma': trial.suggest_float('clf__gamma', 1e-3, 1.0, log=True),
        'clf__kernel': kernel,
    }
    if kernel == 'poly':
        param_grid['clf__degree'] = trial.suggest_int('clf__degree', 2, 5)


    pipeline.set_params(**param_grid)
    results = get_results(FILE,SEEDS)

    values = [r["macro_f1"] for r in results]

    return np.mean(values)





# se obtiene o crea estudio a partir de base de datos sqllite
study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True)

if OPT_TIME > 0: # si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos
    study.optimize(objective, timeout=OPT_TIME)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")
else: # si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada)

    max_n = study.best_params['prepro__post_tfidf__ngram_range_max'] 
    min_n = study.best_params['prepro__post_tfidf__ngram_range_min']

    param_grid = study.best_params # copiamos parámetros desde estudio
    param_grid.pop('prepro__post_tfidf__ngram_range_max') # max y min no existen realmente en tfidf, los quitamos
    param_grid.pop('prepro__post_tfidf__ngram_range_min')
    
    param_grid['prepro__post_tfidf__ngram_range'] = (min_n, max_n)

    pipeline.set_params(**param_grid) # cargamos mejores parámetros
    results = get_results(FILE,SEEDS)
    
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)

    metric_names = ['macro_f1', 'balanced_accuracy', 'matthews_corrcoef', 'roc_auc_ovr']
    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")