import optuna
from lightgbm import LGBMClassifier
import warnings
import numpy as np
from common import get_results,optimize,test_study,make_vectorizer,get_params
import sys

if __name__ == "__main__":

    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    OPT_TIME = params["LGBM"]["OPT_TIME"]
    OPT_STUDY = params["LGBM"]["OPT_STUDY"]
    OPT_DB = params["LGBM"]["OPT_DB"]
    MODEL_PATH = params["LGBM"]["MODEL_PATH"]
    SEEDS = params["COMMON"]["SEEDS"]
    FILE = params["COMMON"]["FILE"]+"/rb_db"
    USE_TEST = params["LGBM"]["USE_TEST"]
    SAVE_MODEL = params["LGBM"]["SAVE_MODEL"]
    NO_TRAIN = params["LGBM"]["NO_TRAIN"]
    LOCAL_MODEL_DIR = params["LGBM"]["LOCAL_MODEL_DIR"]
    ##

    model = LGBMClassifier(
            objective='binary',
            verbose=-1,
        )

    # Función para buscar hiperparámetros.
    def objective(trial):

        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', -0, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        }
        
        if param_grid['subsample_freq'] > 0: # subsample viene condicionado por subsample_freq.
            param_grid['subsample'] = trial.suggest_float('subsample', 0.4, 1.0)

        model.set_params(**param_grid) # Actualizamos hiperparámetros.
        # Se obtienen las métricas por semilla.
        results, _ = get_results(model,FILE,SEEDS,is_test=False,trial=trial) 

        values = [r["macro_f1"] for r in results]

        return np.mean(values)


    warnings.filterwarnings("ignore", message="X does not have valid feature names") # Filtra advertencias.

    # Se obtiene o crea estudio a partir de base de datos SQLite.
    study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True) 

    if OPT_TIME > 0: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos.
        optimize(study,objective,OPT_TIME)
    else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada).
        if NO_TRAIN:
            models = test_study(study,model,FILE,SEEDS,USE_TEST,SAVE_MODEL,LOCAL_MODEL_DIR,NO_TRAIN=NO_TRAIN)
        else:
            models = test_study(study,model,FILE,SEEDS,USE_TEST,SAVE_MODEL,MODEL_PATH)
