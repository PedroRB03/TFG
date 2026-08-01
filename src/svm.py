import optuna
from sklearn import svm
import numpy as np
from common import get_results, optimize, test_study, make_vectorizer, get_params
import sys
import joblib
from pathlib import Path

if __name__ == "__main__":
   
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    OPT_TIME = params["SVM"]["OPT_TIME"]
    OPT_STUDY = params["SVM"]["OPT_STUDY"]
    OPT_DB = params["SVM"]["OPT_DB"]
    MODEL_PATH = params["SVM"]["MODEL_PATH"]
    SEEDS = params["COMMON"]["SEEDS"]
    FILE = params["COMMON"]["FILE"]+"/rb_db"
    USE_TEST = params["SVM"]["USE_TEST"]
    SAVE_MODEL = params["SVM"]["SAVE_MODEL"]
    ##


    model = svm.NuSVC(
        decision_function_shape="ovr",
        probability=True
    )


    # Función para buscar hiperparámetros
    def objective(trial):
        #max_n = trial.suggest_int('pre__tfidf__ngram_range_max', 2, 5)
        #min_n = trial.suggest_int('pre__tfidf__ngram_range_min', 1, 2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])

        param_grid = {
            'nu': trial.suggest_float('nu', 0.1, 0.5),
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
            'kernel': kernel,
        }
        if kernel == 'poly':
            param_grid['degree'] = trial.suggest_int('degree', 2, 5)

        model.set_params(**param_grid)
        results , _ = get_results(model,FILE,SEEDS)

        values = [r["macro_f1"] for r in results]

        return np.mean(values)




    # Se obtiene o crea estudio a partir de base de datos sqllite
    study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True)


    if OPT_TIME > 0: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos
        optimize(study,objective,OPT_TIME)
    else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada)
        models = test_study(study,model,FILE,SEEDS,USE_TEST)
        if SAVE_MODEL:
            print("Guardando modelo SVM...")
            for seed in SEEDS:
                Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
                joblib.dump(models[seed], MODEL_PATH+"/svm_model"+str(seed)+".pkl")
            print("Modelo guardado.")