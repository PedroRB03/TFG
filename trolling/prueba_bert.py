import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_auc_score)
from scipy.special import softmax
df = pd.read_pickle("rb_db.pkl")

RANDOM_STATE = 42

set_seed(RANDOM_STATE)

df_clase_0 = df[df['label'] == 0]
df_clase_1 = df[df['label'] == 1]

df_clase_0 = df_clase_0.iloc[:2495] #2495
df_clase_1 = df_clase_1.iloc[:2495]
df_equilibrado = pd.concat([df_clase_0, df_clase_1])
df_equilibrado = df_equilibrado.sample(frac=1).reset_index(drop=True)


results = []
model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_fast=True,
                                            extra_special_tokens=['[URL]','[USER]']
                                            )

def tokenize_function(examples):
    return tokenizer(examples["txt"], padding="max_length", truncation=True, max_length=256)


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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(skf.split(df_equilibrado['txt'], df_equilibrado['label'])):
    print(f"\n--- Entrenando Fold {fold + 1} ---")
    
    train_ds = Dataset.from_pandas(df_equilibrado.iloc[train_idx][['txt', 'label']])
    val_ds = Dataset.from_pandas(df_equilibrado.iloc[val_idx][['txt', 'label']])
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"./resultados_fold_{fold}",
        eval_strategy='epoch',
        save_strategy='epoch',
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

print("\n" + "="*30)
print("RESULTADOS FINALES CV")
print("="*30)

metric_names = ['eval_macro_f1', 'eval_balanced_accuracy', 'eval_matthews_corrcoef', 'eval_roc_auc_ovr']
for m in metric_names:
    values = [r[m] for r in results]
    print(f"{m[5:]}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")