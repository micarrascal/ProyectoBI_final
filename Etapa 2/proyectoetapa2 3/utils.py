from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

import re
# librería Natural Language Toolkit, usada para trabajar con textos
import nltk
# Punkt permite separar un texto en frases.
nltk.download('punkt')

# stopwords contiene palabras que no aportan información al texto
nltk.download('stopwords')
from nltk.corpus import stopwords


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_cleaned = X.apply(self.clean_text)
        print('X_cleaned: ', X_cleaned)
        return X_cleaned

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)  # Eliminar caracteres especiales
        text = text.lower()  # Convertir a minúsculas
        stop_words = set(stopwords.words('spanish'))  # Definir las stopwords
        text = ' '.join(word for word in text.split() if word not in stop_words)  # Eliminar stopwords
        text = ' '.join(set(text.split()))  # Eliminar palabras repetidas
        return text
