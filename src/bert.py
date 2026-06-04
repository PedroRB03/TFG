
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, logging as tr_log
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax
import torch
from common import balance

## PARÁMETROS
SEEDS = [600,601,602,603,604] # semillas que se usarán
FILE = "data/pkls/crisp/rb_db" # prefijo de archivos a usar si FUZZY = False
FILE_F = "data/pkls/fuzzy/rbf_db" # prefijo de archivos a usar si FUZZY = True
RESULTS = "data/bert_results" # donde guardar mejores modelos
FUZZY = True # Cambia de clasificación a regresión si está en True
EARLY_STOP = 3 # Paciencia del early stopping
BALANCING = "RUS" # Puede ser None (Ninguna ténica de balanceo), RUS (RandomUnderSampler) o ENN (EditedNearestNeighbours) (No usar SMOTE)
##


tr_log.set_verbosity_error() # Quitamos avisos y dejamos solo alertas de errores

results = [] # Resultados de cada seed

model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            extra_special_tokens=['[URL]','[USER]']
                                            )

def tokenize_function(examples):
    return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)

# Función para calcular métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if FUZZY:
        probs = logits.squeeze()
        predictions= np.round(probs).astype(int)
        labels = np.round(labels).astype(int)
        predictions = np.clip(predictions, 0, 1) # por si acaso, aseguramos el rango [0,1]
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


u_file = FILE
if FUZZY:
    u_file = FILE_F

for seed in SEEDS:
    print(f"\n--- Entrenando Seed {seed} ---")
    
    train_ds = Dataset.from_pandas(pd.read_pickle(u_file+str(seed)+"train.pkl"))
    eval_ds = Dataset.from_pandas(pd.read_pickle(u_file+str(seed)+"eval.pkl"))
    test_ds = Dataset.from_pandas(pd.read_pickle(u_file+str(seed)+"test.pkl"))
    
    t_train = train_ds.map(tokenize_function, batched=True)
    t_eval = eval_ds.map(tokenize_function, batched=True)
    t_test = test_ds.map(tokenize_function, batched=True)

    if BALANCING is not None and BALANCING != "SMOTE":
        idxs = np.arange(len(t_train)).reshape(-1, 1)
        labels = np.array(t_train["label"])
        new_idx,_ = balance(idxs,labels,BALANCING,seed)
        new_idx = new_idx.flatten()
        t_train = t_train.select(new_idx)

    if FUZZY:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,torch_dtype=torch.float32)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,torch_dtype=torch.float32)
    
    # Hiperparámetros
    training_args = TrainingArguments(
        output_dir=RESULTS+"/seed"+str(seed),
        eval_strategy='epoch',
        save_strategy='best',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=100,
        seed=seed,
        metric_for_best_model='macro_f1',
        load_best_model_at_end=True,
    )
    
    callbacks = []

    if EARLY_STOP > 0: # Si EARLY_STOP no es cero, incluimos el callback
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=t_train,
        eval_dataset=t_eval,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    
    eval_stats = trainer.evaluate(eval_dataset=t_test)
    results.append(eval_stats)

# Hacemos print de resultados finales
print("\n" + "="*30)
print("RESULTADOS FINALES")
print("="*30)

metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr']
for m in metric_names:
    values = [r[m] for r in results]
    print(f"{m[5:]}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")