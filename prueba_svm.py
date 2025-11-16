import os
import json
import pandas as pd
import optuna
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# --- 1. Cargar el JSON ---

if not os.path.exists("tab_db.json"):
    with open("tab_db.json", "r", encoding="utf-8") as f:
        data = json.load(f)


# --- 2. Preparamos preprocesos y vectorizador ---

tfidf_p = TfidfVectorizer(
    max_features=40588,    
    ngram_range=(3,5),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'post')
    ],
    remainder='drop'  # ignora el resto de columnas
)

# --- 3. Configuramos modelo y pipeline

model = svm.SVC(
    decision_function_shape="ovo",
    class_weight="balanced"
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

def objective(trial):
    # === Hiperparámetros a optimizar ===
    params = {
        'prepro__post_tfidf__max_features': trial.suggest_int('max_features',1000,50000),
        'prepro__post_tfidf__ngram_range': (trial.suggest_int('ngram_range',1,3),trial.suggest_int('ngram_range',1,5))
    }

    # Actualiza el modelo dentro del pipeline
    pipeline.set_params(**params)

    # === Validación cruzada ===
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    # Devuelve la métrica promedio (Optuna maximiza por defecto si usas direction='maximize')
    return scores.mean()


# --- 4. Separamos datos y entrenamos

#print(pipeline.get_params().keys())
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=100)
#print("Mejores hiperparámetros:", study.best_params)
#print("Mejor puntuación F1:", study.best_value)

X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

pipeline.fit(X_train, y_train)
# --- 5. Evaluación ---
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
