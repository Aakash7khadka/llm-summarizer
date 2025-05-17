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


def vectorize_tfidf(corpus: List[str], max_features: int = 5000) -> Tuple:
    """
    Vectorize text using TF-IDF.

    Args:
        corpus (List[str]): List of input text strings.
        max_features (int): Maximum number of features to keep.

    Returns:
        Tuple:
            - vectors: TF-IDF sparse matrix
            - vectorizer: Fitted TfidfVectorizer instance
    """
    logging.info("🔢 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(corpus)
    logging.info(f"TF-IDF shape: {vectors.shape}")
    return vectors, vectorizer


def vectorize_doc2vec(
    corpus: List[str],
    vector_size: int = 100,
    epochs: int = 40,
    min_count: int = 1
) -> Tuple:
    """
    Vectorize text using Doc2Vec.

    Args:
        corpus (List[str]): List of input documents.
        vector_size (int): Dimensionality of the vectors.
        epochs (int): Number of training iterations.
        min_count (int): Minimum word frequency threshold.

    Returns:
        Tuple:
            - vectors: List of document vectors
            - model: Trained Doc2Vec model
    """
    logging.info("📘 Vectorizing with Doc2Vec...")
    tagged_docs = [TaggedDocument(words=doc.split(), tags=[i])
                   for i, doc in enumerate(corpus)]

    model = Doc2Vec(
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        epochs=epochs
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count,
                epochs=model.epochs)

    vectors = [model.infer_vector(doc.words) for doc in tagged_docs]
    logging.info(
        f"Doc2Vec completed: {len(vectors)} vectors of size {vector_size}")
    return vectors, model


def vectorize_bert(corpus: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Tuple:
    """
    Vectorize text using Sentence-BERT.

    Args:
        corpus (List[str]): List of input text strings.
        model_name (str): Name of HuggingFace model to load.

    Returns:
        Tuple:
            - vectors: List of dense embeddings
            - model: Loaded SentenceTransformer
    """
    logging.info(f"🤖 Vectorizing with Sentence-BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    vectors = model.encode(corpus, show_progress_bar=True)
    logging.info(f"BERT embeddings generated: {len(vectors)} vectors")
    return vectors, model
