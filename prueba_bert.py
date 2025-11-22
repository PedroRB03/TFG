import os
import json
import pandas as pd
import optuna
import time
from sklearn import svm
import torch
import numpy as np
from wordcloud import WordCloud
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from lightgbm import LGBMClassifier

# --- 1. Cargar el JSON ---

if not os.path.exists("tab_db.json"):
    with open("tab_db.json", "r", encoding="utf-8") as f:
        data = json.load(f)


# --- 2. Preparamos preprocesos y vectorizador ---
torch.backends.cudnn.benchmark = True

class DebertaEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="microsoft/mdeberta-v3-base", device=None,batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        torch.backends.cudnn.benchmark = True

    def encode(self, texts):
        embs = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)

            cls_batch = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls_batch)

        return np.vstack(embs)


encoder = DebertaEmbeddings()

X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

X_train_emb = encoder.encode(X_train["post"].astype(str).tolist())
X_test_emb  = encoder.encode(X_test["post"].astype(str).tolist())

# --- 3. Configuramos modelo y pipeline
model = LGBMClassifier(
    n_estimators=182,
    learning_rate=0.06,
    subsample=0.5939,
    min_child_samples=30,
    num_leaves=2,
    force_col_wise=True,
    colsample_bytree=0.962,
    class_weight='balanced',
    objective='multiclass',
    num_class=3,
    max_depth=1
)

model2 = svm.NuSVC(
    nu=.5,
    decision_function_shape="ovr",
    class_weight='balanced',
    break_ties=True,
    gamma=0.45
)




# --- 4. Separamos datos y entrenamos


#print(pipeline.get_params().keys())
#study = optuna.create_study(direction="maximize")
#study.optimize(lambda trial:
#    cross_val_score(
#        LGBMClassifier(
#            n_estimators=trial.suggest_int("n_estimators", 100, 800),
#            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#            num_leaves=trial.suggest_int("num_leaves", 8, 256),
#            max_depth=trial.suggest_int("max_depth", -1, 12),
#            objective="multiclass",
#            num_class=3,
#            class_weight="balanced",
#            subsample=trial.suggest_float("subsample", 0.5, 1.0),
#            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
#        ),
#        X_train_emb,
#        y_train,
#        cv=3,
#        scoring="f1_weighted",
#        n_jobs=-1    # ahora sí
#    ).mean(),
#    n_trials=100
#)
#print("Mejores hiperparámetros:", study.best_params)
#print("Mejor puntuación F1:", study.best_value)

t = time.perf_counter()
model.fit(X_train_emb, y_train)
print("Entrenamiento: "+str(time.perf_counter()-t))
# --- 5. Evaluación ---
t = time.perf_counter()
y_pred = model.predict(X_test_emb)
print("Test: "+str(time.perf_counter()-t))

print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
