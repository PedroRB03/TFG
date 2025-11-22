import os
import json
import pandas as pd
import numpy as np
import re
import time
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

if __name__ == "__main__":

    if not os.path.exists("tab_db.json"):
        with open("tab_db.json", "r", encoding="utf-8") as f:
            data = json.load(f)

    def normalize_text(text):
        text = re.sub(r'http\S+', '[URL]', text)
        text = re.sub(r'@\w+', '[USER]', text)
        return text



    tokz = AutoTokenizer.from_pretrained(
                "microsoft/mdeberta-v3-base", 
                use_fast=False
                )
    data_collator = DataCollatorWithPadding(tokenizer=tokz, padding="max_length", max_length=256, return_tensors="pt")

    def tokenize_fn(x):
        return tokz(x['text'], padding=True, truncation=True, max_length=256)

    X_train = pd.read_pickle("X_train.pkl")
    X_test = pd.read_pickle("X_test.pkl")
    y_train = pd.read_pickle("y_train.pkl")
    y_test =pd.read_pickle("y_test.pkl")

    X_train['post']=X_train['post'].apply(normalize_text)
    X_test['post']=X_test['post'].apply(normalize_text)

    ds_train = pd.concat([X_train, y_train],axis=1)
    ds_test = pd.concat([X_test, y_test],axis=1)

    ds_train = Dataset.from_pandas(ds_train)
    ds_test = Dataset.from_pandas(ds_test)

    ds_train = ds_train.rename_column("post", "text")
    ds_train = ds_train.rename_column("ragescore", "labels")
    ds_train = ds_train.remove_columns("__index_level_0__")


    ds_test = ds_test.rename_column("post", "text")
    ds_test = ds_test.rename_column("ragescore", "labels")
    ds_test = ds_test.remove_columns("__index_level_0__")



    tok_train = ds_train.map(tokenize_fn,batched=True)
    tok_test = ds_test.map(tokenize_fn,batched=True)

    tok_train = tok_train.with_format("torch")
    tok_test  = tok_test.with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/mdeberta-v3-base', 
        num_labels=3, 
        dtype=torch.float16
    )

    #model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="bert_ft_opt",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
        data_collator=data_collator
    )



    t = time.perf_counter()
    trainer.train()
    print("Entrenamiento: "+str(time.perf_counter()-t))

    trainer.save_model("modelo_final")
    tokz.save_pretrained("modelo_final")

    print("Modelo guardado")
    # --- Evaluación ---
    t = time.perf_counter()
    predictions = trainer.predict(tok_test)
    y_pred = np.argmax(predictions.predictions,axis=1)
    print("Test: "+str(time.perf_counter()-t))

    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
