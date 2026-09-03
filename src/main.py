from argparse import ArgumentParser
from common import get_params
import pandas as pd
from pathlib import Path
import re
import joblib
import warnings
from transformers import AutoModelForSequenceClassification, Trainer
from bertcore import get_tokenizer
import numpy as np
from datasets import Dataset
from scipy.special import softmax
from transformers.utils import logging as hf_logging
from sklearn.metrics import f1_score,balanced_accuracy_score, matthews_corrcoef, accuracy_score
import time

# Sustituye URLs y menciones a usuarios por tokens correspondientes. Elimina formateo de texto como tipo <b><\b> y &amp; .
def light_normalize_text(text):
    text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs.
    text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios.
    text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)','', text)
    return text

# Aplica función sigmoide para normalización.
def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

if __name__ == "__main__":
    ## PARÁMETROS
    parser = ArgumentParser(prog="main",usage='%(prog)s [input] [options]',description="Toma la ruta a un archivo csv o pkl con un mensaje por fila y devuelve la respuesta de cada modelo especificado a los mensajes.")
    
    parser.add_argument("input",help='Ruta al archivo csv o pkl con un mensaje por fila.')
    parser.add_argument("-m","--model",required=True,help="Modelo a utilizar.",choices=['lgbm','svm','bert','all'])
    parser.add_argument("-c","--config",default='params.ini',help="Especifica ruta al archivo de configuración.")
    parser.add_argument("-l","--limit",default='',help="Limita el número de muestras a las n primeras.")
    parser.add_argument("-p","--proba",action='store_true',help="Muestra el grado de pertenencia a la clase ganadora.")
    parser.add_argument("-t","--time",action='store_true',help="Muestra tiempo tardado en clasificar por muestra y medio.")

    args = parser.parse_args()
    params = get_params(args.config)

    SHOW_LOAD_REPORT = params["BERT"]["SHOW_LOAD_REPORT"]

    lgbm_vfile = params["LGBM"]["VFILE"]
    lgbm_file = params["LGBM"]["FILE"]

    svm_vfile = params["SVM"]["VFILE"]
    svm_file = params["SVM"]["FILE"]

    bertv_file = params["BERT"]["MODEL_NAME"]
    bert_file = params["BERT"]["LOCAL_MODEL_NAME"]
    ##

    f_path = Path(args.input)

    ## VALIDACIÓN DEL INPUT
    if not f_path.is_file():
        print("Archivo no encontrado.")
        exit(0)
    else:
        ext = f_path.suffix.lower()
        if ext not in ['.csv','.pkl']:
            print("Formato no soportado, solo se admite csv o pkl.")
            exit(0)
    if not args.model:
        print("Tiene que especificar qué modelo utilizar {lgbm,svm,bert,all}.")
        exit(0)
    ##
    
    ## FORMACIÓN DEL DATAFRAME
    if ext == '.csv':
        df = pd.read_csv(f_path, header=None)
        df.columns = ['txt']

    else:
        df = pd.read_pickle(f_path)
        df=df.rename(columns={df.columns[0]:"txt"})

    if len(df) == 0:
        print("El conjunto de datos está vacío.")
        exit(0)

    df=df[["txt"]]

    df["txt"] = df["txt"].apply(lambda x: light_normalize_text(str(x)))
    
    if args.limit != '':
        df = df.head(int(args.limit))
    ##

    ## CARGA DE MODELOS
    lgbm=None
    lgbm_v=None
    svm=None
    svm_v=None
    bert=None
    bert_v=None
                    
    if args.model in ["lgbm","all"]:
        lgbm = joblib.load(lgbm_file)
        lgbm_v = joblib.load(lgbm_vfile)
        warnings.filterwarnings("ignore", message="X does not have valid feature names") # Filtra advertencias.
    if args.model in ["svm","all"]:
        svm = joblib.load(svm_file)
        svm_v = joblib.load(svm_vfile)
    if args.model in ["bert","all"]:

        if not SHOW_LOAD_REPORT: # Ocultar mensaje de carga del modelo.
            hf_logging.set_verbosity_error()
        bert = AutoModelForSequenceClassification.from_pretrained(bert_file)
        _,bert_v = get_tokenizer(bertv_file) 

    ##

    ## OBTENCIÓN DE PREDICCIONES

    # Obtenemos predicciones para modelos Scikit.
    def _get_r(model,vec):
        ret = None
        if model:
            msgv = vec.transform(df["txt"]) # Vectorizamos.

            st = time.time_ns()
            p = model.predict_proba(msgv) # Obtenemos probabilidades.
            t = time.time_ns()-st

            p_w = np.max(p,axis=1)
            r = np.argmax(p, axis=1)
            ret = (r,p_w,t)

        return ret

    lgbm_r = _get_r(lgbm,lgbm_v)
    svm_r = _get_r(svm,svm_v)

    # Predicciones de DeBERTaV3.
    bert_r = None
    if bert:
        trainer = Trainer(model=bert)

        bdf = Dataset.from_pandas(df[['txt']])
        msgv = bdf.map(bert_v, batched=True) # Tokenización por lotes.
        st = time.time_ns()
        out = trainer.predict(msgv)
        t = time.time_ns()-st
        logits = out.predictions

        if logits.shape[1] > 1: # Para crisp.
            st = time.time_ns()
            probs = softmax(logits, axis=-1)
            r = np.argmax(logits, axis=-1)
            p = np.max(logits, axis=-1)
            t+= time.time_ns()-st

            bert_r = (r,p,t)
        else: # Para fuzzy.
            st = time.time_ns()
            probs = logits.squeeze()
            r= np.round(probs).astype(int) # Redondeo para obtener 0 o 1.
            r = np.clip(r, 0, 1) # Aseguramos rango [0,1].
            p = sigmoid(probs)
            t+= time.time_ns()-st

            bert_r = (r,p,t) # Se utiliza la función sigmoide para colocar logits entre 0 y 1.
    ##


    ## MOSTRAR RESULTADOS POR PANTALLA
    def _nominal(i):
        if i == 0:
            return "Normal"
        else:
            return "Rage-bait"

    def _showres(res,l,i):
        if res:
            r = res[0][i]
            if args.proba: # Si se quiere mostrar probabilidades.
                    print(f" {l}: ({_nominal(r)},{res[1][i]*100:.2f}%)",end="")
            else:
                    print(f" {l}: ({_nominal(r)})",end="")
    
    def _showtime(res,l):
        if res:
            print(f" {l}: ({res[2]/1e6:.2f}ms total, {res[2]/1e6/len(df):.2f}ms por muestra estimado)",end="")

    print("="*30 + "RESULTADOS" + "="*30)

    for i,msg in enumerate(df["txt"]):
        print(f"{i}: [",end="")
        _showres(lgbm_r,"lgbm",i)
        _showres(svm_r,"svm",i)
        _showres(bert_r,"deberta",i)
        print(" ]")
    if args.time:
        print(f"Tiempo de clasificación: ")
        print(f"    [",end="")
        _showtime(lgbm_r,"lgbm")
        _showtime(svm_r,"svm")
        _showtime(bert_r,"deberta")
        print(" ]")
    ##