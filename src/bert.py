import optuna
import numpy as np
from common import optimize, get_params
from bertcore import *
import sys
from transformers.utils import logging as hf_logging

if __name__ == "__main__":
   
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    OPT_TIME = params["BERT"]["OPT_TIME"]
    OPT_STUDY = params["BERT"]["OPT_STUDY"]
    OPT_DB = params["BERT"]["OPT_DB"]
    SEEDS = params["COMMON"]["SEEDS"]
    FILE = params["COMMON"]["FILE"]+"/rb_db"
    FILE_F = params["COMMON"]["FILE_F"]+"/rbf_db"
    RESULTS = params["BERT"]["RESULTS"]
    FUZZY = params["BERT"]["FUZZY"]
    EARLY_STOP = params["BERT"]["EARLY_STOP"]
    USE_TEST = params["BERT"]["USE_TEST"]
    SAVE_MODEL = params["BERT"]["SAVE_MODEL"]
    BALANCED_CW = params["BERT"]["BALANCED_CW"]
    SHOW_LOAD_REPORT = params["BERT"]["SHOW_LOAD_REPORT"]
    FUZZY_BAL_CRISP = params["COMMON"]["FUZZY_BAL_CRISP"]
    MODEL_NAME = params["BERT"]["MODEL_NAME"]
    NO_TRAIN = params["BERT"]["NO_TRAIN"]
    LOCAL_MODEL_NAME = params["BERT"]["LOCAL_MODEL_NAME"]
    LOCAL_MODEL_DIR = params["BERT"]["LOCAL_MODEL_DIR"]
    ##
   


    # Calculamos ruta de los datos a usar en función del parámetro FUZZY
    u_file = FILE 
    if FUZZY:
        u_file = FILE_F

    # Función para buscar hiperparámetros
    def objective(trial: optuna.Trial):

        param_grid = {
            'learning_rate' : trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            'num_train_epochs' : trial.suggest_int("num_train_epochs", 1 , 4),
            'warmup_steps' : trial.suggest_int("warmup_steps", 100 , 600),
            'weight_decay' : trial.suggest_int("weight_decay", 0.01 , 0.1),
            'gradient_accumulation_steps' : trial.suggest_categorical("gradient_accumulation_steps",[1,2,4,8])
        }
        
        results, _ = trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST=False,SAVE_MODEL=False,BALANCED_CW=BALANCED_CW,FUZZY_BAL_CRISP=FUZZY_BAL_CRISP,MODEL_NAME=MODEL_NAME,trial=trial,tmodel_name=MODEL_NAME)
        
        values = [r["eval_macro_f1"] for r in results]
        
        return np.mean(values)

    if not SHOW_LOAD_REPORT: # Ocultar mensaje de carga del modelo
        hf_logging.set_verbosity_error()


    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    # Se obtiene o crea estudio a partir de base de datos sqllite
    study = optuna.create_study(direction='maximize',study_name=OPT_STUDY,storage=OPT_DB,load_if_exists=True, pruner=pruner)


    if OPT_TIME > 0 and not NO_TRAIN: # Si se busca optimizar, parte del estudio creado y busca por OPT_TIME segundos
        optimize(study,objective,OPT_TIME)
    else: # Si no, obtiene los mejores parámetros hasta el momento (la base de datos debe contener unos mejores valores, no debe ser recién creada)
        if NO_TRAIN:
            testbert(study,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW,FUZZY_BAL_CRISP,model_name=LOCAL_MODEL_DIR,NO_TRAIN=True,tmodel_name=MODEL_NAME)
        else:
            testbert(study,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW,FUZZY_BAL_CRISP,model_name=MODEL_NAME,tmodel_name=MODEL_NAME)
