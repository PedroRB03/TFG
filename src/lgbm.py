import optuna
from lightgbm import LGBMClassifier
import warnings
import numpy as np
from common import get_results,optimize,test_study,make_vectorizer,get_params
import sys
import joblib
from pathlib import Path

if __name__ == "__main__":

    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    OPT_TIME = params["LGBM"]["OPT_TIME"]
    OPT_STUDY = params["LGBM"]["OPT_STUDY"]
    OPT_DB = params["LGBM"]["OPT_DB"]
    MODEL_PATH = params["SVM"]["MODEL_PATH"]
    SEEDS = params["COMMON"]["SEEDS"]
    FILE = params["COMMON"]["FILE"]+"/rb_db"
    USE_TEST = params["LGBM"]["USE_TEST"]
    SAVE_MODEL = params["LGBM"]["SAVE_MODEL"]
    ##

    model = LGBMClassifier(
            objective='binary',
            verbose=-1
        )

    # Función para buscar hiperparámetros
    def objective(trial):
        #max_n = trial.suggest_int('pre__tfidf__ngram_range_max', 2, 5)
        #min_n = trial.suggest_int('pre__tfidf__ngram_range_min', 1, 2)

        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        }

        model.set_params(**param_grid)
        results, _ = get_results(model,FILE,SEEDS)

        values = [r["macro_f1"] for r in results]

        return np.mean(values)


    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    # Se obtiene o crea estudio a partir de base de datos sqllite
    study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True) 

    if OPT_TIME > 0: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos
        optimize(study,objective,OPT_TIME)
    else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada)
        models = test_study(study,model,FILE,SEEDS,USE_TEST)
        if SAVE_MODEL:
            print("Guardando modelo LGBM...")
            for seed in SEEDS:
                Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
                joblib.dump(models[seed], MODEL_PATH+"/lgbm_model"+str(seed)+".pkl")
            print("Modelo guardado.")