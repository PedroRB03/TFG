
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax
import torch
from common import get_class_weights
import shutil
import os
from pathlib import Path
import torch.nn.functional as F
from transformers import set_seed

model_name = "microsoft/mdeberta-v3-base"
    
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            extra_special_tokens=['[URL]','[USER]']
                                            )

# Tokeniza los ejemplares del dataset para su posterior uso por el modelo bert
def tokenize_function(examples):
    return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)

# Función para calcular métricas
def compute_metrics(FUZZY, eval_pred):
    logits, labels = eval_pred

    if FUZZY:
        probs = logits.squeeze()
        predictions= np.round(probs).astype(int)
        labels = np.round(labels).astype(int)
        predictions = np.clip(predictions, 0, 1) # Por si acaso, aseguramos el rango [0,1]
        roc_auc = roc_auc_score(labels, probs) 

    else:
        probs = softmax(logits, axis=-1)
        predictions = np.argmax(logits, axis=-1)
        roc_auc = roc_auc_score(labels, probs[:, 1]) 
        
    return {
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'matthews_corrcoef': matthews_corrcoef(labels, predictions),
        'roc_auc_ovr': roc_auc
    }

# Permite entrenar el modelo con los parámetros dados y muestra una media de las métricas de evaluación
def testbert(study,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW=False,FUZZY_BAL_CRISP=False):
    
    param_grid = study.best_params # Cargamos mejores parámetros

    results = trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL,BALANCED_CW,FUZZY_BAL_CRISP)

    # Hacemos print de resultados finales
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)
    metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr']
    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m[5:]}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

# Carga y aplica la función de tokenización a los datasets de entrenamiento, evaluación y test y los devuelve.
def obtain_tokenized(u_file, seed):
    train_ds = pd.read_pickle(u_file+str(seed)+"train.pkl")
    eval_ds = pd.read_pickle(u_file+str(seed)+"eval.pkl")
    test_ds = pd.read_pickle(u_file+str(seed)+"test.pkl")

    # Eliminamos columnas con texto pasado por TF-IDF
    train_ds.drop(columns=["vec"], inplace=True)
    eval_ds.drop(columns=["vec"], inplace=True)
    test_ds.drop(columns=["vec"], inplace=True)

    train_ds = Dataset.from_pandas(train_ds)
    eval_ds = Dataset.from_pandas(eval_ds)
    test_ds = Dataset.from_pandas(test_ds)
    
    t_train = train_ds.map(tokenize_function, batched=True)
    t_eval = eval_ds.map(tokenize_function, batched=True)
    t_test = test_ds.map(tokenize_function, batched=True)

    return t_train, t_eval, t_test

# Función para sobrescribir el método compute_loss y aplicar pesos de clase
def weighted_compute_loss(class_weights, outputs, labels, num_items_in_batch=None):
    logits = outputs.logits

    # Fuzzy (num_labels == 1)
    if logits.shape[-1] == 1:

        label_idx = torch.bucketize( # Calculamos a qué peso de clase le corresponde cada ejemplar
            labels,
            boundaries=torch.tensor([0.165, 0.495, 0.83], device=labels.device) # [(0+0.33)/2,(0.33+0.66)/2,(0.66+1)/2]
            # Usamos intervalos para asignar pesos de clase ya que tenemos valores continuos
        )
        sample_weights = class_weights.to(labels.device)[label_idx]
        
        loss = F.mse_loss(logits.squeeze(), labels.squeeze(),reduction="none")
        loss = (loss * sample_weights).mean()

    # Crisp 
    else:
        loss = F.cross_entropy(logits, labels,weight=class_weights.to(logits.device))

    return loss

# Entrena el modelo con las semillas y parámetros dados y devuelve las métricas de evaluación
def trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL=False,BALANCED_CW=False,FUZZY_BAL_CRISP=False):

    results = [] # Resultados de cada semilla

    for seed in SEEDS:
        print(f"\n--- Entrenando Seed {seed} ---")

        set_seed(seed) 

        t_train, t_eval, t_test = obtain_tokenized(u_file,seed) # datasets tokenizados

        dname = RESULTS+"/seed"+str(seed)
        Path(dname).mkdir(parents=True, exist_ok=True) # Creamos ruta si no existe

        shutil.rmtree(dname) # Borramos directorio y archivos donde se guardan los checkpoints
        os.mkdir(dname) # Lo volvemos a crear

        if FUZZY:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,dtype=torch.float32) # Para regresión
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,dtype=torch.float32)
        model.resize_token_embeddings(len(tokenizer))
        
        # Hiperparámetros
        training_args = TrainingArguments(
            output_dir=dname,
            eval_strategy='epoch',
            save_strategy='best',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=param_grid['learning_rate'],
            num_train_epochs=param_grid['num_train_epochs'],
            warmup_steps=param_grid['warmup_steps'],
            weight_decay=param_grid['weight_decay'],
            logging_steps=100,
            seed=seed,
            data_seed=seed,
            metric_for_best_model='macro_f1',
            load_best_model_at_end=True,
            #full_determinism=True, # viene bien para determinadas pruebas pero hace que el entrenamiento sea MUY lento
        )
        
        callbacks = []

        if EARLY_STOP > 0: # Si EARLY_STOP no es cero, incluimos el callback
            callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)]

        if BALANCED_CW: # Calculamos pesos de clase
            class_weights = get_class_weights(t_train["label"], FUZZY_BAL_CRISP) 
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            
        def _c_loss(outputs, labels, num_items_in_batch=None):
            return weighted_compute_loss(class_weights,outputs,labels,num_items_in_batch)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train,
            eval_dataset=t_eval,
            compute_metrics=lambda eval_pred: compute_metrics(FUZZY, eval_pred),
            callbacks=callbacks,
            compute_loss_func=None if not BALANCED_CW else _c_loss
        )

        trainer.train()
        if SAVE_MODEL: # Guardamos modelo entrenado
            sdname = RESULTS+"/bert_model"+str(seed)
            Path(sdname).mkdir(parents=True, exist_ok=True)
            print("Guardando modelo BERT...")
            trainer.save_model(sdname)
            print("Modelo guardado.")
        

        if USE_TEST: # Calcula métricas con datos de test o evaluación
            eval_stats = trainer.evaluate(eval_dataset=t_test)
        else:
            eval_stats = trainer.evaluate(eval_dataset=t_eval)
        results.append(eval_stats)

    return results

