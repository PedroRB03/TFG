import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
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
    max_depth=1
)

pipeline = Pipeline([
    ('prepro', preprocessor),
    ('clf', model)
])

def objective(trial):
    # === Hiperparámetros a optimizar ===
    ngrams = [(1,2),(1,3),(2,3),(2,4),(2,5)]
    params = {
        'clf__num_leaves': trial.suggest_int('num_leaves', 5, 255),
        'clf__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8, log=True),
        'clf__n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'clf__subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'clf__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'clf__max_depth': trial.suggest_int('max_depth', -1, 12),
        'clf__min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'prepro__post_tfidf__max_features': trial.suggest_int('max_features',1000,50000),
        'prepro__post_tfidf__ngram_range': ngrams[trial.suggest_int('ngram',0,len(ngrams)-1)]
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

    # Devuelve la métrica promedio (Optuna maximiza por defecto si usas direction='maximize')
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

pipeline.fit(X_train, y_train)
# --- 5. Evaluación ---
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average='macro')
probs = pipeline.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
mcc = matthews_corrcoef(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print("\nMétricas adicionales:")
print(f"Macro F1: {macro_f1:.4f}")
print(f"ROC-AUC (OvR): {roc_auc}")
print(f"MCC: {mcc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")