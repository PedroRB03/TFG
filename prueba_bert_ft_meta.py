import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report, 
    confusion_matrix
)
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModel, AutoConfig
from datasets import Dataset
from dataclasses import dataclass

class CustomCLSModel(nn.Module):
    def __init__(self, model_name, num_labels, extra_features_dim=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        hidden = self.config.hidden_size
        self.classifier = nn.Linear(hidden + extra_features_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, extra_features=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # shape (batch, hidden)
        
        # Concatenar características extra
        if extra_features is None:
            raise ValueError("Debe proporcionarse 'extra_features' al modelo")
        
        x = torch.cat([cls_emb, extra_features], dim=1)
        
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

@dataclass
class DataCollatorWithFeatures:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        extra = [item["extra_features"] for item in batch]
        for item in batch:
            del item["extra_features"]
        
        enc = self.tokenizer.pad(batch, return_tensors="pt")
        enc["extra_features"] = torch.tensor(extra, dtype=torch.float32)
        return enc
    
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        extra = inputs.pop("extra_features")
        outputs = model(**inputs, extra_features=extra)

        logits = outputs["logits"]

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
if __name__ == "__main__":
        
    X_train = pd.read_pickle("X_train.pkl")
    X_test = pd.read_pickle("X_test.pkl")
    y_train = pd.read_pickle("y_train.pkl")
    y_test =pd.read_pickle("y_test.pkl")


    #base = "modelo_final_bueno"
    base = "microsoft/mdeberta-v3-base"

    tokz = AutoTokenizer.from_pretrained(
                base, 
                use_fast=False
                )

    def tokenize_fn(x):
        return tokz(x['input'], padding='max_length', truncation=True, max_length=256)


    ds_train = pd.concat([X_train, y_train],axis=1)
    ds_test = pd.concat([X_test, y_test],axis=1)

    ds_train = Dataset.from_pandas(ds_train)
    ds_test = Dataset.from_pandas(ds_test)

    ds_train = ds_train.rename_column("post", "input")
    ds_train = ds_train.rename_column("ragescore", "labels")
    ds_train = ds_train.remove_columns("__index_level_0__")


    ds_test = ds_test.rename_column("post", "input")
    ds_test = ds_test.rename_column("ragescore", "labels")
    ds_test = ds_test.remove_columns("__index_level_0__")

    def add_features(ds):
        ds["extra_features"] = [ds["upvote_ratio"], ds["score"]]
        return ds

    ds_train = ds_train.map(add_features)
    ds_test = ds_test.map(add_features)

    tok_train = ds_train.map(tokenize_fn,batched=True)
    tok_test = ds_test.map(tokenize_fn,batched=True)

    dds = tok_train.train_test_split(test_size=0.25)
    train_ds = dds['train']
    eval_ds  = dds['test']

  
    model = CustomCLSModel(
        model_name=base, 
        num_labels=3,
        extra_features_dim=2
    )

    #model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="bert_ft_opt",
        learning_rate=3e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        fp16=False,
        label_smoothing_factor=0.1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    class_counts = np.bincount(y_train)
    total = len(y_train)

    weights = total / (len(class_counts) * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    print("Class weights:", weights)
    
    collator = DataCollatorWithFeatures(tokenizer=tokz)
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        class_weights=weights,
        data_collator=collator
    )
    
    
    macro_f1 = 0
    roc_auc = 0
    mcc = 0
    bal_acc = 0
    N=20

    for i in range(0,N):
            
        TRAIN = True
        if TRAIN:
            t = time.perf_counter()
            trainer.train()
            #trainer.save_model("modelo_final")
            #tokz.save_pretrained("modelo_final")
            print("Entrenamiento: "+str(time.perf_counter()-t))
            print("Modelo guardado")


        # --- Evaluación ---
 
        t = time.perf_counter()
        predictions = trainer.predict(tok_test)
        y_pred = np.argmax(predictions.predictions,axis=1)
        print(y_pred)
        print("Test: "+str(time.perf_counter()-t))

        #print(classification_report(y_test, y_pred, zero_division=0))
        #print("Matriz de confusión:")
        #print(confusion_matrix(y_test, y_pred))

        macro_f1 += f1_score(y_test, y_pred, average='macro')

        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        roc_auc += roc_auc_score(y_test, probs, multi_class='ovr')

        mcc += matthews_corrcoef(y_test, y_pred)
        bal_acc += balanced_accuracy_score(y_test, y_pred)

    print("\nMétricas adicionales:")
    print(f"Macro F1: {macro_f1/N:.4f}")
    print(f"ROC-AUC (OvR): {roc_auc/N}")
    print(f"MCC: {mcc/N:.4f}")
    print(f"Balanced Accuracy: {bal_acc/N:.4f}")