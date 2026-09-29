from argparse import ArgumentParser
from common import get_params, normalize_text
import pandas as pd
from pathlib import Path
import joblib
import warnings
from transformers import AutoModelForSequenceClassification
from bertcore import get_tokenizer
from transformers.utils import logging as hf_logging
import inference

if __name__ == "__main__":
    ## PARÁMETROS
    parser = ArgumentParser(prog="main",usage='%(prog)s [input] [options]',description="Toma la ruta a un archivo csv o pkl con un mensaje por fila y devuelve la respuesta de cada modelo especificado a los mensajes.")
    
    parser.add_argument("input",help='Ruta al archivo csv o pkl con un mensaje por fila.')
    parser.add_argument("-m","--model",required=True,help="Modelo a utilizar.",choices=['lgbm','svm','bert','all'])
    parser.add_argument("-c","--config",default='params.ini',help="Especifica ruta al archivo de configuración.")
    parser.add_argument("-p","--proba",action='store_true',help="Muestra la salida cruda del modelo respecto a la clase ganadora.")
    parser.add_argument("-t","--time",action='store_true',help="Muestra tiempo tardado cada modelo en calcular las predicciones.")

    parser.add_argument("-l","--limit",default='',help="Limita el número de muestras a las n primeras.")
    
    args = parser.parse_args()
    params = get_params(args.config)

    SHOW_LOAD_REPORT = params["BERT"]["SHOW_LOAD_REPORT"]

    lgbm_vfile = params["LGBM"]["VFILE"]
    lgbm_file = params["LGBM"]["FILE"]

    svm_vfile = params["SVM"]["VFILE"]
    svm_file = params["SVM"]["FILE"]

    bertv_path = "microsoft/deberta-v3-small"
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

    else:
        df = pd.read_pickle(f_path)


    df = df[df.columns[0]].to_frame() # Se deja solo la primera columna
    df.columns = ['txt']

    if len(df) == 0:
        print("El conjunto de datos está vacío.")
        exit(0)

    df=df[["txt"]]

    df["txt"] = df["txt"].apply(lambda x: normalize_text(str(x),light=True))
    
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
        _,bert_v = get_tokenizer(bertv_path) 

    ##

    ## OBTENCIÓN DE PREDICCIONES
    lgbm_r = inference.lgbm_pred(lgbm,lgbm_v,df)
    svm_r = inference.svm_pred(svm,svm_v,df)
    bert_r = inference.bert_pred(bert,bert_v,df)
    ##


    ## MOSTRAR RESULTADOS POR PANTALLA
    def _nominal(i):
        if i == 0:
            return "Normal"
        else:
            return "Trolling"

    def _showres(res,l,i):
        if res:
            r = res[0][i]
            if args.proba: # Si se quiere mostrar probabilidades.
                    print(f" {l}: ({_nominal(r)},{res[1][i]:.2f})",end="")
            else:
                    print(f" {l}: ({_nominal(r)})",end="")
    
    def _showtime(res,l):
        if res:
            print(f" {l}: ({res[2]/1e6:.2f}ms total)",end="")

    # Se muestran predicciones.
    for i,msg in enumerate(df["txt"]):
        print(f"{i}: [",end="")
        _showres(lgbm_r,"lgbm",i)
        _showres(svm_r,"svm",i)
        _showres(bert_r,"deberta",i)
        print(" ]")
    # Se muestran tiempos.
    if args.time:
        print(f"Tiempo de inferencia: ")
        print(f"[",end="")
        _showtime(lgbm_r,"lgbm")
        _showtime(svm_r,"svm")
        _showtime(bert_r,"deberta")
        print(" ]",end="")
    ##