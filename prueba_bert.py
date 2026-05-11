import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax


## PARÁMETROS
SEEDS = [600,601,602,603,604] # semillas que se usarán
FILE = "pkls/rbf_db" # prefijo de archivos a usar
FUZZY = True # Cambia de clasificación a regresión si está en True
EARLY_STOP = 1 # Paciencia del early stopping
##


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


for seed in SEEDS:
    print(f"\n--- Entrenando Seed {seed} ---")
    
    train_ds = Dataset.from_pandas(pd.read_pickle(FILE+str(seed)+"train.pkl"))
    eval_ds = Dataset.from_pandas(pd.read_pickle(FILE+str(seed)+"eval.pkl"))
    test_ds = Dataset.from_pandas(pd.read_pickle(FILE+str(seed)+"test.pkl"))
    
    t_train = train_ds.map(tokenize_function, batched=True)
    t_eval = eval_ds.map(tokenize_function, batched=True)
    t_test = test_ds.map(tokenize_function, batched=True)

    if FUZZY:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    
    # Hiperparámetros
    training_args = TrainingArguments(
        #output_dir=f"./resultados_fold_{seed}",
        eval_strategy='epoch',
        save_strategy='no',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-6,
        num_train_epochs=3,
        warmup_steps=900,
        weight_decay=0.01,
        logging_steps=100,
        seed=seed,
        metric_for_best_model='macro_f1',
        load_best_model_at_end=True,
    )

    if EARLY_STOP == 0: # Si EARLY_STOP es cero, no incluimos el callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train,
            eval_dataset=t_eval,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train,
            eval_dataset=t_eval,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP)] # Incluimos early stopping si no es 0
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