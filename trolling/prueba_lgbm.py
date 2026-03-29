import pandas as pd
import optuna
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import warnings
import numpy as np

# --- 2. Preparamos preprocesos y vectorizador ---

tfidf_p = TfidfVectorizer(
    ngram_range=(1,3),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'text'),
#        ('nums', 'passthrough', ['ragescore'])
    ],
    remainder='drop'  # ignora el resto de columnas
)
# --- 3. Configuramos modelo y pipeline

#{'clf__n_estimators': 380, 'clf__learning_rate': 0.04211511334177564, 'clf__num_leaves': 26, 'clf__max_depth': 11, 
# 'clf__min_child_samples': 9, 'clf__subsample': 0.6712777519334666, 'clf__colsample_bytree': 0.9305218497444706}

model = LGBMClassifier(
    n_estimators=380,
    learning_rate=0.04211511334177564,
    subsample=0.6712777519334666,
    min_child_samples=9,
    num_leaves=26,
    colsample_bytree=0.9305218497444706,
    class_weight='balanced',
    objective='multiclass',
    num_class=2,
    max_depth=-1,
    verbose=-1
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

def objective(trial):
    # 1. Definimos el espacio de búsqueda de hiperparámetros
    param_grid = {
        'clf__n_estimators': trial.suggest_int('clf__n_estimators', 100, 1000),
        'clf__learning_rate': trial.suggest_float('clf__learning_rate', 0.01, 0.3, log=True),
        'clf__num_leaves': trial.suggest_int('clf__num_leaves', 20, 150),
        'clf__max_depth': trial.suggest_int('clf__max_depth', 3, 12),
        'clf__min_child_samples': trial.suggest_int('clf__min_child_samples', 5, 100),
        'clf__subsample': trial.suggest_float('clf__subsample', 0.4, 1.0),
        'clf__colsample_bytree': trial.suggest_float('clf__colsample_bytree', 0.4, 1.0),
    }

    # 2. Actualizamos el pipeline con los parámetros sugeridos
    # Usamos set_params para no reconstruir todo el objeto
    pipeline.set_params(**param_grid)

    # 3. Ejecutamos la validación cruzada
    # Usamos f1_macro como métrica objetivo (puedes cambiarla)
    score = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
    
    return score.mean()

# --- 4. Separamos datos y entrenamos
ALL = pd.read_pickle("rb_db.pkl")

X = ALL.drop(columns="ragescore")
y = ALL["ragescore"]

OPTIMIZAR = False

warnings.filterwarnings("ignore", message="X does not have valid feature names")

if OPTIMIZAR:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Prueba con 50-100 iteraciones

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