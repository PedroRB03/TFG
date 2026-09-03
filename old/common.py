
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Lo de abajo estaba en balance
#rng = np.random.default_rng(seed)
#cdf = df[df['label'] == y_i].reset_index(drop=True)
#txt_rng = cdf['txt'][rng.integers(0,len(cdf))] # añadimos un texto aleatorio de relleno, solo utilizado en caso de usar BERT.

# Crea un pipeline de scikit-learn con el modelo pasado, usado por lgbm y svm
def make_pipeline(model):
    tfidf_p = TfidfVectorizer( # Vectorizador
        #ngram_range=(1,3),    
        stop_words="english",
        lowercase=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
        ('tfidf', tfidf_p, 'txt')
        ],
        remainder='drop'  
    )

    pipeline = Pipeline([
        ('pre', preprocessor), # Vectorización
        ('clf', model) # Clasificador
    ]) 
    return pipeline

# no usada, quitar en versión final
def remNcheckDupes(df):
    print(f"Duplicados CRISP normalizado: {len(df['txt'])-len(df['txt'].drop_duplicates())}")
    df = df.drop_duplicates(subset=['txt']) # Quitamos duplicados
    print(f"Duplicados CRISP barajado: {len(df['txt'])-len(df['txt'].drop_duplicates())}")