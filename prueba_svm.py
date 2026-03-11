import pandas as pd
import optuna
import warnings
import numpy as np
from sklearn import set_config
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report, 
    confusion_matrix
)

# --- 2. Preparamos preprocesos y vectorizador ---

tfidf_p = TfidfVectorizer(
    max_features=23838,    
    ngram_range=(1,2),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'post'),
        ('nums', StandardScaler(), ['upvote_ratio','score'])
    ],
    remainder='drop'  # ignora el resto de columnas
)
# --- 3. Configuramos modelo y pipeline

model = svm.NuSVC(
    nu=.5,
    decision_function_shape="ovr",
    class_weight='balanced',
    break_ties=True,
    gamma=0.45,
    probability=True
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])


# --- 4. Separamos datos y entrenamos
ALL = pd.read_pickle("all_df.pkl")

X = ALL.drop(columns="ragescore")
y = ALL["ragescore"]

scores = cross_validate(pipeline,X,y,cv=5,scoring=["f1_macro","roc_auc_ovr","matthews_corrcoef","balanced_accuracy"])


print(scores)
print("MEANS:")
for key, values in scores.items():
    if key.startswith("test_"):
        print(f"{key}: {np.mean(values):.4f}")
