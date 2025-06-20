import logging
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


def vectorize_tfidf(train_corpus: List[str], test_corpus: List[str], max_features: int = 5000) -> Tuple:
    """
    Vectorize text using TF-IDF.

    Args:
        train_corpus (List[str]): List of input train text strings.
        test_corpus (List[str]): List of input test text strings.
        max_features (int): Maximum number of features to keep.

    Returns:
        Tuple:
            - Train vector
            - Test vector
            - vectorizer: Fitted TfidfVectorizer instance
    """
    logging.info("🔢 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_vectors = vectorizer.fit_transform(train_corpus)
    test_vectors = vectorizer.transform(test_corpus)
    return train_vectors, test_vectors, vectorizer


def train_doc2vec(
    train_docs: List[str],
    vector_size: int = 100,
    epochs: int = 40,
    min_count: int = 2
) -> Doc2Vec:
    """
    Vectorize text using Doc2Vec.

    Args:
        train_docs (List[str]): List of input train text strings.
        vector_size (int): Dimensionality of the vectors.
        epochs (int): Number of training iterations.
        min_count (int): Minimum word frequency threshold.

    Returns:
        - model: A Doc2Vec model
    """
    logging.info("📘 Doc2Vec Model...")
    tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)])
                   for i, doc in enumerate(train_docs)]

    model = Doc2Vec(
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=3,
        epochs=epochs
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model


def vectorize_doc2vec (model, docs: List[str]) -> List[List[float]]:
    """
        Generate document vectors using a trained Doc2Vec model.

        Args:
            model (Doc2Vec): A trained Gensim Doc2Vec model.
            docs (List[str]): List of raw text documents to be vectorized.

        Returns:
            List[List[float]]: List of document vectors, where each vector corresponds to a document.
    """
    vectors = [model.infer_vector(doc.split()) for doc in docs]
    logging.info(
        f"Doc2Vec completed: {len(vectors)} vectors of size {model.vector_size}")
    return vectors


def vectorize_bert(train_corpus: List[str], test_corpus: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Tuple:
    """
    Vectorize text using Sentence-BERT.

    Args:
        train_corpus (List[str]): List of input train text strings.
        test_corpus (List[str]): List of input test text strings.
        model_name (str): Name of HuggingFace model to load.

    Returns:
        Tuple:
            - Train vector
            -Test vector
            - model: Loaded SentenceTransformer
    """
    logging.info(f"🤖 Vectorizing with Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    train_vectors = model.encode(train_corpus, show_progress_bar=True)
    test_vectors = model.encode(test_corpus, show_progress_bar=True)
    logging.info(f"BERT embeddings generated: {len(train_vectors)} vectors")
    return train_vectors, test_vectors, model
