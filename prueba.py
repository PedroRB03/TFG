import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

COMM_LIMIT=20

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
        data["post"].append(dcc[k]["title"].replace(';','')+" ; "+dcc[k]["selftext"].replace(';',''))
        data["upvote_ratio"].append(dcc[k]["upvote_ratio"])
        data["score"].append(dcc[k]["score"])
        i=0
        for c in dcc[k]["comments"]:
            if i >= COMM_LIMIT:
                break
            bod = dcc[k]["comments"][c]["body"].replace(';','')
            if not pattern.match(bod) and len(bod) > 0:
                i+=1
                comms+=bod+" ; "
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

tfidf = TfidfVectorizer(
    max_features=20000,    # puedes ajustar según tamaño del corpus
    ngram_range=(1,2),     # unigramas y bigramas
    stop_words="english"   # elimina stopwords en ingles
)

preprocessor = ColumnTransformer(
    transformers=[
        ('post_tfidf', tfidf, 'post'),
        ('comments_tfidf', tfidf, 'comments'),
        ('numeric','passthrough',['upvote_ratio','score'])
    ],
    remainder='drop'  # ignora el resto de columnas
)

# --- 3. Configuramos modelo y pipeline

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    random_state=42
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

# --- 4. Separamos datos y entrenamos

X_train, X_test, y_train, y_test = train_test_split(
    df[['post', 'comments','upvote_ratio','score']],
    df['ragescore'],
    test_size=0.25,
    random_state=42,
    stratify=df['ragescore']
)

pipeline.fit(X_train, y_train)

# --- 5. Evaluación ---
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))