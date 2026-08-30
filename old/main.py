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

# Sustituye URLs y menciones a usuarios por tokens correspondientes. Elimina formateo de texto como tipo <b><\b> y &amp;
def light_normalize_text(text):
    text = re.sub(r'(\[.*\]\(.*\))|(<URL>)|(http[^\s]+)|(https[^\s]+)', '[URL]', text) # Formateo de URLs
    text = re.sub(r'(@\w+)|(<USER>)', '[USER]', text) # Formateo usuarios
    text = re.sub(r'(\&amp\;)|(\<b\>)|(\<b\\\/\>)|(\<\\\/b\>)','', text)
    return text

# Aplica función sigmoide para normalización
def sigmoid(arr):
    return 1/(1 + np.exp(-arr))

if __name__ == "__main__":
    ## PARÁMETROS
    parser = ArgumentParser(prog="main",usage='%(prog)s [input] [options]',description="Toma la ruta a un archivo csv o pkl con un mensaje por fila y devuelve la respuesta de cada modelo especificado a los mensajes.")
    
    parser.add_argument("input",help='Ruta al archivo csv o pkl con un mensaje por fila.')
    parser.add_argument("-m","--model",required=True,help="Modelo a utilizar.",choices=['lgbm','svm','bert','all'])
    parser.add_argument("-c","--config",default='params.ini',help="Especifica ruta al archivo de configuración.")
    parser.add_argument("-p","--proba",action='store_true',help="Muestra el grado de pertenencia a la clase ganadora.")
    parser.add_argument("-s","--stats",action='store_true',help=
    """
    Si hay una segunda columna con las clases en el input, 
    muestra: Macro-F1, Precisión Balanceada, Coeficiente de Correlación de Matthews y Precisión.
    """
    )

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

    ## Validación del input
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
    
    ## FORMACIÓ DEL DATAFRAME
    if ext == '.csv':
        df = pd.read_csv(f_path, header=None)
        if len(df.columns) > 1:
            df.columns = ['txt','labels']
        else:
            df.columns = ['txt']

    else:
        df = pd.read_pickle(f_path)
        if len(df.columns) > 1:
            df=df.rename(columns={df.columns[0]:"txt",df.columns[1]:"labels"})
        else:
            df=df.rename(columns={df.columns[0]:"txt"})

    if len(df) == 0:
        print("El conjunto de datos está vacío.")
        exit(0)

    # Si el dataset viene con la respuesta de antemano se deja esa columna
    if len(df.columns) > 1:
        df=df[["txt","labels"]]
    else:
        df=df[["txt"]]

    df["txt"] = df["txt"].apply(lambda x: light_normalize_text(str(x)))
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
        warnings.filterwarnings("ignore", message="X does not have valid feature names") # Filtra advertencias
    if args.model in ["svm","all"]:
        svm = joblib.load(svm_file)
        svm_v = joblib.load(svm_vfile)
    if args.model in ["bert","all"]:

        if not SHOW_LOAD_REPORT: # Ocultar mensaje de carga del modelo
            hf_logging.set_verbosity_error()
        bert = AutoModelForSequenceClassification.from_pretrained(bert_file)
        _,bert_v = get_tokenizer(bertv_file) 

    ##

    ## OBTENCIÓN DE PREDICCIONES

    # Obtenemos predicciones para modelos scikit
    def _get_r(model,vec):
        ret = None
        if model:
            msgv = vec.transform(df["txt"])
            p = model.predict_proba(msgv)

            p_w = np.max(p,axis=1)
            r = np.argmax(p, axis=1)

            if args.proba:
                ret = (r,p_w)
            else:
                ret = (r,)
        return ret

    lgbm_r = _get_r(lgbm,lgbm_v)
    svm_r = _get_r(svm,svm_v)

    # Predicciones de DeBERTaV3
    bert_r = None
    if bert:
        trainer = Trainer(model=bert)

        bdf = Dataset.from_pandas(df[['txt']])
        msgv = bdf.map(bert_v, batched=True)
        out = trainer.predict(msgv)
        logits = out.predictions

        if logits.shape[1] > 1:
            probs = softmax(logits, axis=-1)
            r = np.argmax(logits, axis=-1)

            if args.proba:
                bert_r = (r,np.max(logits, axis=-1))
            else:
                bert_r = (r,)
        else:
            probs = logits.squeeze()
            r= np.round(probs).astype(int)
            r = np.clip(r, 0, 1)

            if args.proba:
                bert_r = (r,sigmoid(probs)) # Se utiliza la función sigmoide para colocar logits entre 0 y 1
            else:
                bert_r = (r,)
    ##


    ## MOSTRAR RESULTADOS POR PANTALLA
    def _nominal(i):
        if i == 0:
            return "Normal"
        else:
            return "Rage-bait"

    def _showres(res,l,scores):
        if res:
            r = res[0][i]
            scores[r]+=1
            if args.proba:
                print(f" {l}: ({_nominal(r)},{res[1][i]*100:.2f}%)",end="")
            else:
                print(f" {l}: ({_nominal(r)})",end="")
        return scores

    print("="*30 + "RESULTADOS" + "="*30)
    p_score = []

    for i,msg in enumerate(df["txt"]):
        print(f"{i}: [",end="")
        scores = [0,0]
        _showres(lgbm_r,"lgbm",scores)
        _showres(svm_r,"svm",scores)
        _showres(bert_r,"deberta",scores)

        if args.model == "all":
            pop = np.argmax(scores)
            p_score.append(pop)
            print(f" popular: ({_nominal(pop)})",end="")

        print(" ]")

    if len(p_score) > 0 and args.stats and len(df.columns) > 1:
        print("="*20 + "MOSTRANDO ESTADÍSTICAS" + "="*20)
        print(f"macro_f1: {f1_score(df['labels'].astype(int), p_score, average='macro'):.4f}")
        print(f"balanced_accuracy: {balanced_accuracy_score(df['labels'].astype(int), p_score):.4f}")
        print(f"matthews_corrcoef: {matthews_corrcoef(df['labels'].astype(int), p_score):.4f}")
        print(f"accuracy: {accuracy_score(df['labels'].astype(int), p_score):.4f}")
  
    ##