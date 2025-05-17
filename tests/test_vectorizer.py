from vectorization.vectorizers import vectorize_tfidf, vectorize_doc2vec, vectorize_bert

# Use longer and more diverse sample texts to avoid training errors in Doc2Vec
sample_texts = [
    "Machine learning and natural language processing are exciting fields of study.",
    "Doc2Vec builds vector representations from tagged documents using context and semantics.",
    "This sentence contains diverse and sufficient vocabulary to ensure proper training works."
]


def test_vectorize_tfidf():
    vecs, model = vectorize_tfidf(sample_texts)
    assert vecs.shape[0] == len(sample_texts)
    assert vecs.shape[1] > 0


def test_vectorize_doc2vec():
    vecs, model = vectorize_doc2vec(
        sample_texts, vector_size=50, epochs=20, min_count=1)
    assert len(vecs) == len(sample_texts)
    assert all(isinstance(v, list) or hasattr(v, "__len__") for v in vecs)
    assert len(vecs[0]) > 0


def test_vectorize_bert():
    vecs, model = vectorize_bert(sample_texts)
    assert len(vecs) == len(sample_texts)
    assert len(vecs[0]) > 0
