import optuna
from lightgbm import LGBMClassifier
import warnings
import numpy as np
from common import get_results,optimize,test_study,make_vectorizer

## PARÁMETROS
OPT_TIME = 60 # Dejar en 0 si no se quieren buscar hiperparámetros
OPT_STUDY = "lgbm-study" # Nombre del estudio de optuna
OPT_DB = "sqlite:///results/lgbm-study.db" # Nombre de la base de datos con estudio
SEEDS = [600,601,602,603,604] # Semillas que se usarán
FILE = "data/pkls/crisp/rb_db" # Prefijo de archivos a usar
USE_TEST = True # Usar conjunto de test cuando no se quiere optimizar
BALANCING = "SMOTE" # Puede ser None (Ninguna ténica de balanceo), RUS (RandomUnderSampler), ENN (EditedNearestNeighbours) o SMOTE
##

model = LGBMClassifier(
        objective='binary',
        verbose=-1
    )


# Función para buscar hiperparámetros
def objective(model,trial):
    max_n = trial.suggest_int('pre__tfidf__ngram_range_max', 2, 5)
    min_n = trial.suggest_int('pre__tfidf__ngram_range_min', 1, 2)

    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
    }

    tfidf = make_vectorizer((min_n,max_n))
    model.set_params(**param_grid)
    results = get_results(model,FILE,SEEDS,tfidf,BALANCING)

    values = [r["macro_f1"] for r in results]

    return np.mean(values)


warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Se obtiene o crea estudio a partir de base de datos sqllite
study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True) 

if OPT_TIME > 0: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos
    optimize(study,model,objective,OPT_TIME)
else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada)
    test_study(study,model,FILE,SEEDS,USE_TEST,BALANCING)