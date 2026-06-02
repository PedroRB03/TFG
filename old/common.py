
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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