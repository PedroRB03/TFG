
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax
import torch
from common import balance,get_params
import shutil
import os
from pathlib import Path


model_name = "microsoft/mdeberta-v3-base"
    
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            extra_special_tokens=['[URL]','[USER]']
                                            )

def tokenize_function(examples):
    return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)

# Función para calcular métricas
def compute_metrics(FUZZY, eval_pred):
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


def testbert(study,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL):
    # Hacemos print de resultados finales
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)

    param_grid = study.best_params

    results = trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL)

    metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr']
    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m[5:]}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")


def obtain_tokenized(u_file, seed):
    train_ds = pd.read_pickle(u_file+str(seed)+"train.pkl")
    eval_ds = pd.read_pickle(u_file+str(seed)+"eval.pkl")
    test_ds = pd.read_pickle(u_file+str(seed)+"test.pkl")

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

def trainbert(param_grid,u_file,SEEDS,FUZZY,EARLY_STOP,RESULTS,USE_TEST,SAVE_MODEL=False):

    results = [] # Resultados de cada seed

    for seed in SEEDS:
        print(f"\n--- Entrenando Seed {seed} ---")

        
        t_train, t_eval, t_test = obtain_tokenized(u_file,seed)
        dname = RESULTS+"/seed"+str(seed)
        Path(dname).mkdir(parents=True, exist_ok=True)

        shutil.rmtree(dname)
        os.mkdir(dname)

        if FUZZY:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,dtype=torch.float32)
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
            compute_metrics=lambda eval_pred: compute_metrics(FUZZY, eval_pred),
            callbacks=callbacks
        )

        trainer.train()
        if SAVE_MODEL:
            sdname = RESULTS+"/bert_model"+str(seed)
            Path(sdname).mkdir(parents=True, exist_ok=True)
            print("Guardando modelo BERT...")
            trainer.save_model(sdname)
            print("Modelo guardado.")
        

        if USE_TEST:
            eval_stats = trainer.evaluate(eval_dataset=t_test)
        else:
            eval_stats = trainer.evaluate(eval_dataset=t_eval)
        results.append(eval_stats)

    return results
