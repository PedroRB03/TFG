import os
import re
import json
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

COMM_LIMIT=40

# --- 1. Cargar el JSON ---
if not os.path.exists("tab_db.json"):
    dcc = {}
    with open("reddit_db.json", "r", encoding="utf-8") as f:
        j = json.load(f)
        dcc = j["posts"]
    pattern = re.compile(".*(http|\[deleted\]).*")

    data = {
        "post" : [],
        "upvote_ratio" : [],
        "score" : [],
        "comments" : [],
        "ragescore" : []
    }

    for k in dcc:
        comms = ""
        data["post"].append(dcc[k]["title"]+".\n"+dcc[k]["selftext"])
        data["upvote_ratio"].append(dcc[k]["upvote_ratio"])
        data["score"].append(dcc[k]["score"])
        i=0
        for c in dcc[k]["comments"]:
            if i >= COMM_LIMIT:
                break
            bod = dcc[k]["comments"][c]["body"]
            if not pattern.match(bod) and len(bod) > 0:
                i+=1
                comms+=bod+".\n"
        data["comments"].append(comms)
        data["ragescore"].append(dcc[k]["ragescore"])
    with open("tab_db.json", "w", encoding="utf-8") as f:
        json.dump(data,f)
else:
    with open("tab_db.json", "r", encoding="utf-8") as f:
        data = json.load(f)


df = pd.DataFrame(data)
print(df.head())

# --- 2. Preparamos preprocesos y vectorizador ---

tfidf_p = TfidfVectorizer(
    max_features=40588,    
    ngram_range=(1,1),    
    stop_words="english",
    lowercase=True
)

tfidf_c = TfidfVectorizer(
    max_features=40588,    
    ngram_range=(1,1),    
    stop_words="english",
    lowercase=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf_p, 'post'),
        ('comments_tfidf', tfidf_c, 'comments'),
        ('numeric','passthrough',['upvote_ratio','score'])
    ],
    remainder='drop'  # ignora el resto de columnas
)

# --- 3. Configuramos modelo y pipeline

model = LGBMClassifier(
    n_estimators=672,
    learning_rate=0.21712964105005444,
    subsample=0.7712662637970672,
    min_child_samples=14,
    num_leaves=110,
    colsample_bytree=0.8546906154792115,
    objective='multiclass',
    num_class=3,
    max_depth=11,
    random_state=42
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

def objective(trial):
    # === Hiperparámetros a optimizar ===
    params = {
        'clf__num_leaves': trial.suggest_int('num_leaves', 5, 255),
        'clf__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8, log=True),
        'clf__n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'clf__subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'clf__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'clf__max_depth': trial.suggest_int('max_depth', -1, 12),
        'clf__min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'prepro__post_tfidf__max_features': trial.suggest_int('max_features',1000,50000),
        'prepro__comments_tfidf__max_features': trial.suggest_int('max_features',1000,50000),
        'prepro__post_tfidf__ngram_range': (1,trial.suggest_int('ngram_range',1,3)),
        'prepro__comments_tfidf__ngram_range': (1,trial.suggest_int('ngram_range',1,3))
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

X_train, X_test, y_train, y_test = train_test_split(
    df[['post', 'comments','upvote_ratio','score']],
    df['ragescore'],
    test_size=0.1,
    random_state=42,
    stratify=df['ragescore']
)
#print(pipeline.get_params().keys())
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=100)
#print("Mejores hiperparámetros:", study.best_params)
#print("Mejor puntuación F1:", study.best_value)

pipeline.fit(X_train, y_train)
# --- 5. Evaluación ---
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
