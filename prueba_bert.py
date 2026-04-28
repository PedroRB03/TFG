import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax


## PARÁMETROS
RANDOM_STATE = 42
FILE = "rb_db.pkl"
##

df = pd.read_pickle(FILE)
set_seed(RANDOM_STATE)

results = [] # Resultados del CV

model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            extra_special_tokens=['[URL]','[USER]']
                                            )

# Tokenizador
def tokenize_function(examples):
    return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)

# Métricas a mostrar durante el entrenamiento
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = softmax(logits, axis=-1)
    predictions = np.argmax(logits, axis=-1)
    
    roc_auc = roc_auc_score(labels, probs[:, 1]) 
    
    return {
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'matthews_corrcoef': matthews_corrcoef(labels, predictions),
        'roc_auc_ovr': roc_auc
    }

# Obtenemos folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Por cada fold
for fold, (train_idx, val_idx) in enumerate(skf.split(df['txt'], df['label'])):
    print(f"\n--- Entrenando Fold {fold + 1} ---")
    
    train_ds = Dataset.from_pandas(df.iloc[train_idx][['txt', 'label']])
    val_ds = Dataset.from_pandas(df.iloc[val_idx][['txt', 'label']])
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    
    # Hiperparámetros
    training_args = TrainingArguments(
        #output_dir=f"./resultados_fold_{fold}",
        eval_strategy='epoch',
        save_strategy='no',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-6,
        num_train_epochs=3,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=100,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    eval_stats = trainer.evaluate()
    results.append(eval_stats)

# Hacemos print de resultados finales
print("\n" + "="*30)
print("RESULTADOS FINALES CV")
print("="*30)

metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr']
for m in metric_names:
    values = [r[m] for r in results]
    print(f"{m[5:]}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")