import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")


def clean_text_light(text: str) -> str:
    """
    Minimal cleaning for LLMs, BERT, or summarization.
    Preserves sentence structure and stopwords.
    """
    text = str(text).strip().replace("\n", " ").lower()
    return text


def clean_text_strict(text: str) -> str:
    """
    Aggressive cleaning for traditional ML models (e.g., TF-IDF, SVM).
    Removes punctuation, stopwords, and applies stemming.
    """
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)
