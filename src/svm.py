import optuna
from sklearn import svm
import numpy as np
from common import get_results, optimize, test_study, make_vectorizer, get_params
import sys
from pathlib import Path

if __name__ == "__main__":
   
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    OPT_TIME = params["SVM"]["OPT_TIME"]
    OPT_STUDY = params["SVM"]["OPT_STUDY"]
    OPT_DB = params["SVM"]["OPT_DB"]
    MODEL_PATH = params["SVM"]["MODEL_PATH"]
    SEEDS = params["COMMON"]["SEEDS"]
    FILE = params["COMMON"]["FILE"]
    USE_TEST = params["SVM"]["USE_TEST"]
    SAVE_MODEL = params["SVM"]["SAVE_MODEL"]
    KERNEL = params["SVM"]["KERNEL"]
    NO_TRAIN = params["SVM"]["NO_TRAIN"]
    LOCAL_MODEL_DIR = params["SVM"]["LOCAL_MODEL_DIR"]
    ##


    model = svm.SVC(
        #decision_function_shape="ovr",
        kernel=KERNEL,
        class_weight="balanced"
    )


    # Función para buscar hiperparámetros.
    def objective(trial):

        param_grid = {'C': trial.suggest_float('C', 1e-3, 1000,log=True)}
        
        match KERNEL:
            case "rbf":
                param_grid['gamma'] = trial.suggest_float('gamma', 1e-4, 10.0, log=True)
            case "poly":
                param_grid['degree'] = trial.suggest_int('degree', 2, 5)
                param_grid['gamma'] = trial.suggest_float('gamma', 1e-4, 10.0, log=True)
                param_grid['coef0'] = trial.suggest_float('coef0', 0.0, 10.0, log=False)
            case "sigmoid":
                param_grid['gamma'] = trial.suggest_float('gamma', 1e-4, 10.0, log=True)
                param_grid['coef0'] = trial.suggest_float('coef0', -10.0, 10.0, log=False)

        model.set_params(**param_grid) # Actualizamos hiperparámetros.
        # Se obtienen las métricas por semilla.
        results ,_ = get_results(model,FILE,SEEDS,is_test=False,trial=trial)

        values = [r["macro_f1"] for r in results]

        return np.mean(values)




    # Se obtiene o crea estudio a partir de base de datos SQLite.
    study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True)


    if OPT_TIME > 0: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos.
        optimize(study,objective,OPT_TIME)
    else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada).
        if NO_TRAIN:
            models = test_study(study,model,FILE,SEEDS,USE_TEST,SAVE_MODEL,LOCAL_MODEL_DIR,NO_TRAIN=NO_TRAIN)
        else:
            models = test_study(study,model,FILE,SEEDS,USE_TEST,SAVE_MODEL,MODEL_PATH)
