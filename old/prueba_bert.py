
import pandas as pd
import optuna
import warnings
import time
from sklearn import svm
import torch
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report, 
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModel
from lightgbm import LGBMClassifier


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
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)

            cls_batch = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls_batch)

        return np.vstack(embs)


encoder = DebertaEmbeddings()

ALL = pd.read_pickle("all_df.pkl")

X = ALL.drop(columns="ragescore")
y = ALL["ragescore"]

X_emb = encoder.encode(X["post"].astype(str).tolist())
num_features_train = X[["upvote_ratio", "score"]].values
X = np.hstack([X_emb, num_features_train])

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
    gamma=0.45,
    probability=True
)

pipeline = Pipeline([
    ('clf', model)
])



# --- 4. Separamos datos y entrenamos




warnings.filterwarnings("ignore", message="X does not have valid feature names")
scores = cross_validate(model,X,y,cv=10,scoring=["f1_macro","roc_auc_ovr","matthews_corrcoef","balanced_accuracy"])


print(scores)
print("MEANS:")
for key, values in scores.items():
    if key.startswith("test_"):
        print(f"{key}: {np.mean(values):.4f}")