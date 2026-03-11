import pandas as pd
import optuna
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import warnings
import numpy as np
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
        ('nums', 'passthrough', ['upvote_ratio','score'])
    ],
    remainder='drop'  # ignora el resto de columnas
)
# --- 3. Configuramos modelo y pipeline

model = LGBMClassifier(
    n_estimators=182,
    learning_rate=0.06,
    subsample=0.5939,
    min_child_samples=30,
    num_leaves=75,
    colsample_bytree=0.962,
    class_weight='balanced',
    objective='multiclass',
    num_class=3,
    max_depth=1,
    verbose=-1
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])


# --- 4. Separamos datos y entrenamos
ALL = pd.read_pickle("all_df.pkl")

X = ALL.drop(columns="ragescore")
y = ALL["ragescore"]

warnings.filterwarnings("ignore", message="X does not have valid feature names")
scores = cross_validate(pipeline,X,y,cv=5,scoring=["f1_macro","roc_auc_ovr","matthews_corrcoef","balanced_accuracy"])


print(scores)
print("MEANS:")
for key, values in scores.items():
    if key.startswith("test_"):
        print(f"{key}: {np.mean(values):.4f}")