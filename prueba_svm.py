import os
import json
import pandas as pd
import optuna
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report, 
    confusion_matrix
)

# --- 1. Cargar el JSON ---

if not os.path.exists("tab_db.json"):
    with open("tab_db.json", "r", encoding="utf-8") as f:
        data = json.load(f)


# --- 2. Preparamos preprocesos y vectorizador ---

tfidf_p = TfidfVectorizer(
    max_features=1980,
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

#model = svm.SVC(
#    decision_function_shape="ovr",
#    class_weight="balanced",
#    break_ties=True,
#    C=192,
#    gamma=0.45
#)

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

def objective(trial):
    # === Hiperparámetros a optimizar ===
    ngrams = [(1,2),(1,3),(2,3),(2,4),(2,5)]
    params = {
        'prepro__post_tfidf__max_features': trial.suggest_int('max_features',1000,50000),
        'prepro__post_tfidf__ngram_range': ngrams[trial.suggest_int('ngram',0,len(ngrams)-1)],
        'clf__C' : trial.suggest_categorical('C', range(160,200,1)),
        'clf__gamma' : trial.suggest_categorical('gamma',[x / 1000.0 for x in range(300,500,25)] ),
        'clf__decision_function_shape' : trial.suggest_categorical('fshape',['ovr','ovo'])
    }

    # Actualiza el modelo dentro del pipeline
    pipeline.set_params(**params)

    # === Validación cruzada ===
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1
    )

    return scores.mean()


# --- 4. Separamos datos y entrenamos

X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

#print(pipeline.get_params().keys())
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=1000)
#print("Mejores hiperparámetros:", study.best_params)
#print("Mejor puntuación F1:", study.best_value)

macro_f1 = 0
roc_auc = 0
mcc = 0
bal_acc = 0
N=100

for i in range(0,N):
    pipeline.fit(X_train, y_train)
    # --- 5. Evaluación ---
    y_pred = pipeline.predict(X_test)

    #print(classification_report(y_test, y_pred))
    #print("Matriz de confusión:")
    #print(confusion_matrix(y_test, y_pred))


    macro_f1 += f1_score(y_test, y_pred, average='macro')

    probs = pipeline.predict_proba(X_test)
    roc_auc += roc_auc_score(y_test, probs, multi_class='ovr')

    mcc += matthews_corrcoef(y_test, y_pred)
    bal_acc += balanced_accuracy_score(y_test, y_pred)

print("\nMétricas adicionales:")
print(f"Macro F1: {macro_f1/N:.4f}")
print(f"ROC-AUC (OvR): {roc_auc/N}")
print(f"MCC: {mcc/N:.4f}")
print(f"Balanced Accuracy: {bal_acc/N:.4f}")