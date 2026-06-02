import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import SMOTE

# balancing puede ser None (Ninguna ténica de balanceo), RUS (RandomUnderSampler), ENN (EditedNearestNeighbours) o SMOTE
def balance(x,y,balancing,seed):
    bal = None
    match balancing:
        case "RUS":
            bal = RandomUnderSampler(random_state=seed)
        case "ENN":
            bal = EditedNearestNeighbours()
        case "SMOTE":
            bal = SMOTE(random_state=seed)

    if bal:
        y = (y*100).round().astype(int)
    
        red_X, red_Y = bal.fit_resample(x,y)
        x = red_X
        y = (red_Y.astype(float)) / 100
    

    return x,y

# Calcular métricas a partir de resultados de un modelo
def compute_metrics(y_test,y_pred,y_probs):
    return {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'roc_auc_ovr': roc_auc_score(y_test, y_probs)
    }

# Obtener métricas a partir de un modelo y distintas semillas
def get_results(model,file,seeds,tfidf,balancing=None,is_test=False):
    results = []

    for seed in seeds:
        train_ds = pd.read_pickle(file+str(seed)+"train.pkl")
        if is_test:
            test_ds = pd.read_pickle(file+str(seed)+"test.pkl")
        else:
            test_ds = pd.read_pickle(file+str(seed)+"eval.pkl")
        
        X_trn = tfidf.fit_transform(train_ds["txt"])
        X_tst = tfidf.transform(test_ds["txt"])
        Y_trn = train_ds["label"]
        Y_tst = test_ds["label"]
        
        if balancing is not None:
            X_trn, Y_trn = balance(X_trn,Y_trn,balancing,seed)

        model.fit(X_trn,Y_trn)
        y_pred = model.predict(X_tst)
        y_probs = model.predict_proba(X_tst)[:,1]
        
        results.append(compute_metrics(Y_tst,y_pred,y_probs))

    return results

# Crea un vectorizador con el rango de ngram dado
def make_vectorizer(ngram_range):
    return TfidfVectorizer( # Vectorizador
        ngram_range=ngram_range,    
        stop_words="english",
        #lowercase=True
    )

# Busca los mejores parámetros para el estudio y pipeline pasado
# Necesita una función objective que acepte el pipeline como primer parámetro
def optimize(study,model,objective,timeout):
    study.optimize(lambda trial: objective(model,trial), timeout=timeout)

    print("--- MEJORES PARÁMETROS ---")
    print(study.best_params)
    print(f"Mejor F1-Macro: {study.best_value:.4f}")

# Usa los mejores parámetros del estudio pasado y evalua el conjunto de datos de test o evaluación
# Dejar use_test en True si se quieren usar los conjuntos de datos de test, dejar en False si se quiere usar los de evaluación
def test_study(study,model,file,seeds,use_test,balancing):
    max_n = study.best_params['pre__tfidf__ngram_range_max'] 
    min_n = study.best_params['pre__tfidf__ngram_range_min']

    param_grid = study.best_params # Copiamos parámetros desde estudio
    param_grid.pop('pre__tfidf__ngram_range_max') # max y min no existen realmente en tfidf, los quitamos
    param_grid.pop('pre__tfidf__ngram_range_min')

    tfidf = make_vectorizer((min_n,max_n))
    model.set_params(**param_grid) # Cargamos mejores parámetros
    results = get_results(model,file,seeds,tfidf,balancing,is_test=use_test)
    
    print("\n" + "="*30)
    print("RESULTADOS FINALES")
    print("="*30)

    metric_names = ['macro_f1', 'balanced_accuracy', 'matthews_corrcoef', 'roc_auc_ovr']
    for m in metric_names:
        values = [r[m] for r in results]
        print(f"{m}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")