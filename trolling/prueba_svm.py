import pandas as pd
import numpy as np
import optuna
from sklearn import set_config
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm


tfidf_p = TfidfVectorizer(
    ngram_range=(1,3),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'txt'),
#        ('nums', 'passthrough', ['ragescore'])
    ],
    remainder='drop'
)

model = svm.NuSVC(
    nu=0.22163740307384033,
    decision_function_shape="ovr",
    class_weight='balanced',
    break_ties=True,
    kernel='linear',
    gamma=0.025777463245085116,
    probability=True
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

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

ALL = pd.read_pickle("rb_db.pkl")

X = ALL.drop(columns="label")
y = ALL["label"]

OPTIMIZAR = False

if OPTIMIZAR:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")

else:
        
    scores = cross_validate(pipeline,X,y,cv=5,scoring=["f1_macro","roc_auc_ovr","matthews_corrcoef","balanced_accuracy"])


    print(scores)
    print("MEANS:")
    for key, values in scores.items():
        if key.startswith("test_"):
            print(f"{key}: {np.mean(values):.4f}")

