from models.trainers import train_model
from vectorization.vectorizers import vectorize_tfidf

sample_texts = [
    "Machine learning is fascinating.",
    "Support vector machines are useful."
]
sample_labels = [0, 1]


def test_train_svm():
    X, vec = vectorize_tfidf(sample_texts)
    model = train_model(X, sample_labels, model_type="svm")
    assert hasattr(model, 'predict')


def test_train_mlp():
    X, vec = vectorize_tfidf(sample_texts)
    model = train_model(X, sample_labels, model_type="mlp")
    assert hasattr(model, 'predict')


def test_train_rf():
    X, vec = vectorize_tfidf(sample_texts)
    model = train_model(X, sample_labels, model_type="rf")
    assert hasattr(model, 'predict')
