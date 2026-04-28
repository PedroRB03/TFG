import pandas as pd
import numpy as np
import optuna
from sklearn import set_config
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm


## PARÁMETROS
OPTIMIZAR = False # Dejar en False si no se quieren buscar hiperparámetros
RANDOM_STATE = 42
CV = 5 # Folds para Cross Validation
FILE = "rb_db.pkl"
##

tfidf_p = TfidfVectorizer( # Vectorizador
    ngram_range=(1,3),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'txt'), # solo vamos a vectorizar
    ],
    remainder='drop'
)

model = svm.NuSVC( # Hiperparámetros
    nu=0.22163740307384033,
    decision_function_shape="ovr",
    #class_weight='balanced',
    break_ties=True,
    kernel='linear',
    gamma=0.025777463245085116,
    probability=True
)

pipeline = Pipeline([
    ('prepro', preprocessor), # vectorización
    ('clf', model) # clasificador
])

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

    score = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
    
    try:
        score = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
        return score.mean()
    except Exception as e:
        print(f"Error en trial: {e}")
        return 0.0

ALL = pd.read_pickle(FILE)

X = ALL.drop(columns="label")
y = ALL["label"]

if OPTIMIZAR:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")

else:
        
    scores = cross_validate(pipeline,X,y,cv=5,scoring=["f1_macro","roc_auc_ovr","matthews_corrcoef","balanced_accuracy"])


    print(scores)
    print("MEANS:") # Print de media de los folds
    for key, values in scores.items():
        if key.startswith("test_"):
            print(f"{key}: {np.mean(values):.4f}")

