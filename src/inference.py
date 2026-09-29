import time
import numpy as np
from datasets import Dataset
from scipy.special import softmax
from transformers import Trainer

# Obtiene las predicciones, salidas crudas y tiempo de inferencia del modelo LGBM pasado con vectorizador lgbm_v
# sobre la columna "txt" del DataFrame df.
def lgbm_pred(lgbm,lgbm_v,df):
    lgbm_r = None
    if lgbm and lgbm_v and df is not None:
        msgv = lgbm_v.transform(df["txt"]) # Vectorizamos.

        st = time.time_ns()
        p = lgbm.predict_proba(msgv) # Obtenemos probabilidades.
        t = time.time_ns()-st

        p_w = np.max(p,axis=1)
        r = np.argmax(p, axis=1)
        lgbm_r = (r,p_w,t)

    return lgbm_r

# Obtiene las predicciones, salidas crudas y tiempo de inferencia del modelo SVM pasado con vectorizador svm_v
# sobre la columna "txt" del DataFrame df.
def svm_pred(svm,svm_v,df):
    svm_r = None
    if svm and svm_v and df is not None:
        msgv = svm_v.transform(df["txt"]) # Vectorizamos.

        st = time.time_ns()
        p = svm.decision_function(msgv) # Obtenemos distancias al hiperplano.
        t = time.time_ns()-st

        r = svm.classes_[(p > 0).astype(int)] # Obtenemos salidas exactas, clase 0 o 1.
        svm_r = (r,p,t)
    return svm_r

# Obtiene las predicciones, salidas crudas y tiempo de inferencia del modelo DeBERTaV3 pasado con vectorizador bert_v
# sobre la columna "txt" del DataFrame df.
def bert_pred(bert,bert_v,df):
    # Predicciones de DeBERTaV3.
    bert_r = None
    if bert and bert_v and df is not None:
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
            probs = logits.flatten()
            r= np.round(probs).astype(int) # Redondeo, no se aceptan valores continuos como respuesta.
            r = np.clip(r, 0, 1) # Aseguramos que sea 0 o 1.
            t+= time.time_ns()-st
            
            bert_r = (r,probs,t) 
    return bert_r