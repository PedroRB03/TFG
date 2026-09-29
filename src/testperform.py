
from common import get_params
import pandas as pd
import joblib
import warnings
from transformers import AutoModelForSequenceClassification
from bertcore import get_tokenizer
from transformers.utils import logging as hf_logging
import inference
import sys
import numpy as np

if __name__ == "__main__":
    ## PARÁMETROS
    params = get_params(sys.argv[1] if len(sys.argv) > 1 else "params.ini")
    seeds = params["COMMON"]["SEEDS"]

    svm_mdir = params["SVM"]["LOCAL_MODEL_DIR"]
    svm_vdir = params["SVM"]["LOCAL_VEC_DIR"]
    lgbm_mdir = params["LGBM"]["LOCAL_MODEL_DIR"]
    lgbm_vdir = params["LGBM"]["LOCAL_VEC_DIR"]

    bertv_path = "microsoft/deberta-v3-small"
    bert_dir = params["BERT"]["LOCAL_MODEL_DIR"]

    file = params["COMMON"]["FILE"]

    SHOW_LOAD_REPORT = params["BERT"]["SHOW_LOAD_REPORT"]

    ##


    times = {"lgbm":[],"svm":[],"bert":[]} # Estructura donde guardar tiempo de inferencia de cada modelo.
    
    for seed in seeds:
        df = pd.read_pickle(file+"/s"+str(seed)+"test.pkl") # Se carga datos de test.

        ## CARGA DE MODELOS
        lgbm_file = lgbm_mdir+"/mseed"+str(seed)+".pkl"
        lgbm_vfile = lgbm_vdir+"/tfidf"+str(seed)+".pkl"
        svm_file = svm_mdir+"/mseed"+str(seed)+".pkl"
        svm_vfile = svm_vdir+"/tfidf"+str(seed)+".pkl"
        bert_file = bert_dir+"/bert_model"+str(seed)

        lgbm = joblib.load(lgbm_file)
        lgbm_v = joblib.load(lgbm_vfile)
        warnings.filterwarnings("ignore", message="X does not have valid feature names") # Filtra advertencias.

    
        svm = joblib.load(svm_file)
        svm_v = joblib.load(svm_vfile)
        
        if not SHOW_LOAD_REPORT: # Ocultar mensaje de carga del modelo.
            hf_logging.set_verbosity_error()

        bert = AutoModelForSequenceClassification.from_pretrained(bert_file)
        _,bert_v = get_tokenizer(bertv_path) 

        ##

        for _ in range(0,5): # Se repite 5 veces cada semilla
            lgbm_r = inference.lgbm_pred(lgbm,lgbm_v,df)
            svm_r = inference.svm_pred(svm,svm_v,df)
            bert_r = inference.bert_pred(bert,bert_v,df)

            times["lgbm"].append(lgbm_r[2]/1e6)
            times["svm"].append(svm_r[2]/1e6)
            times["bert"].append(bert_r[2]/1e6)
        
    print("="*20 + "TIEMPO DE INFERENCIA" + "="*20)
    print(f"lgbm: {np.mean(times["lgbm"]):.4f}ms +/- {np.std(times["lgbm"]):.4f}ms")
    print(f"svm: {np.mean(times["svm"]):.4f}ms +/- {np.std(times["svm"]):.4f}ms")
    print(f"bert: {np.mean(times["bert"]):.4f}ms +/- {np.std(times["bert"]):.4f}ms")

    