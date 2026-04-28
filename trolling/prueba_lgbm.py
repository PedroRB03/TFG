import pandas as pd
import optuna
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import warnings
import numpy as np

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

model = LGBMClassifier(
    n_estimators=924,
    learning_rate=0.10070282202088496,
    subsample=0.4030040988322811,
    min_child_samples=5,
    num_leaves=21,
    colsample_bytree=0.5922499111669536,
    class_weight='balanced',
    objective='multiclass',
    num_class=2,
    max_depth=6,
    verbose=-1
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

def objective(trial):
    max_n = trial.suggest_int('prepro__post_tfidf__ngram_range_max', 2, 5)
    min_n = trial.suggest_int('prepro__post_tfidf__ngram_range_min', 1, 2)
    param_grid = {
        'prepro__post_tfidf__ngram_range': (min_n, max_n),
        'clf__n_estimators': trial.suggest_int('clf__n_estimators', 100, 1000),
        'clf__learning_rate': trial.suggest_float('clf__learning_rate', 0.01, 0.3, log=True),
        'clf__num_leaves': trial.suggest_int('clf__num_leaves', 20, 150),
        'clf__max_depth': trial.suggest_int('clf__max_depth', 3, 12),
        'clf__min_child_samples': trial.suggest_int('clf__min_child_samples', 5, 100),
        'clf__subsample': trial.suggest_float('clf__subsample', 0.4, 1.0),
        'clf__colsample_bytree': trial.suggest_float('clf__colsample_bytree', 0.4, 1.0),
    }

    pipeline.set_params(**param_grid)

    score = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
    
    return score.mean()


ALL = pd.read_pickle("rb_db.pkl")

X = ALL.drop(columns="label")
y = ALL["label"]

OPTIMIZAR = False

warnings.filterwarnings("ignore", message="X does not have valid feature names")

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