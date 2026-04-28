
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


ALL = pd.read_pickle("all_df.pkl")

X = ALL["post"]
y = ALL["ragescore"]

base = "microsoft/mdeberta-v3-base"

tokz = AutoTokenizer.from_pretrained(
            base, 
            use_fast=True
            )

def tokenize_fn(x):
    return tokz(x['input'], padding='max_length', truncation=True, max_length=258)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    roc = roc_auc_score(labels, probs, multi_class="ovr")
    f1 = f1_score(labels, preds, average="macro")
    bal = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    return {
        "roc_auc": roc,
        "macro_f1": f1,
        "balanced_accuracy": bal,
        "mcc" : mcc
    }

args = TrainingArguments(
    output_dir="bert_ft_opt",
    learning_rate=1.5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    fp16=False,
    label_smoothing_factor=0,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    adam_epsilon=1e-6, # Un poco más alto que el default (1e-8)
    max_grad_norm=1.0,  # Importante para evitar gradientes explosivos
    logging_steps=10
)

class_counts = np.bincount(y)
total = len(y)

weights = total / (len(class_counts) * class_counts)
weights = torch.tensor(weights, dtype=torch.float32)
print("Class weights:", weights)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
  


kfold = KFold(n_splits=5, shuffle=True, random_state=42)

X_np = X.to_numpy()
y_np = y.to_numpy()

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_np)):
    print(f"Fold {fold+1}")
    
    # Crear Dataset de Hugging Face
    train_ds = Dataset.from_dict({"input": X_np[train_idx].tolist(), "labels": y_np[train_idx].tolist()})
    val_ds   = Dataset.from_dict({"input": X_np[val_idx].tolist(), "labels": y_np[val_idx].tolist()})
    
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)


    model = AutoModelForSequenceClassification.from_pretrained(
        base, 
        num_labels=3
    )
    # Crear trainer
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        class_weights=weights,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # Evaluar
    metrics = trainer.evaluate()
    fold_results.append(metrics)

# Promediar resultados
avg_metrics = {k: np.mean([fold[k] for fold in fold_results]) for k in fold_results[0]}
print("Promedio cross-validation:", avg_metrics)