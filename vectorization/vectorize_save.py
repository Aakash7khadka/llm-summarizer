from vectorization.vectorizers import vectorize_tfidf, train_doc2vec, vectorize_doc2vec, vectorize_bert
import numpy as np
import pickle
import joblib
from scipy import sparse
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer


def vectorize_full_dataset(train_texts, test_texts, train_labels, test_labels):
    """"
    Vectorizes the full 20 News dataset using TF-IDF, Doc2Vec, and BERT.
    """
    # TF-IDF
    tfidf_train_vecs, tfidf_test_vecs, tfidf_vectorizer = vectorize_tfidf(train_texts, test_texts)

    # Doc2Vec
    doc2vec_model = train_doc2vec(train_texts)
    doc2vec_train_vecs = vectorize_doc2vec(doc2vec_model, train_texts)
    doc2vec_test_vecs = vectorize_doc2vec(doc2vec_model, test_texts)

    # BERT
    bert_train_vecs, bert_test_vecs, bert_model = vectorize_bert(train_texts, test_texts)

    #Save everything
    sparse.save_npz('data/20news/tfidf_train_5000.npz', tfidf_train_vecs)
    sparse.save_npz('data/20news/tfidf_test_5000.npz', tfidf_test_vecs)
    joblib.dump(tfidf_vectorizer, 'data/20news/tfidf_vectorizer_full.pkl')
    np.save('data/20news/train_labels_5000.npy', train_labels)
    np.save('data/20news/test_labels_5000.npy', test_labels)
    np.save('data/20news/doc2vec_train_def.npy', np.array(doc2vec_train_vecs))
    np.save('data/20news/doc2vec_test_def.npy', np.array(doc2vec_test_vecs))
    doc2vec_model.save('data/20news/doc2vec_model_full.model')
    np.save('data/20news/bert_train.npy', bert_train_vecs)
    np.save('data/20news/bert_test.npy', bert_test_vecs)
    bert_model.save('data/20news/bert_sentence_transformer_model_full')


def vectorize_llm_summary(train_summaries_llm, test_summaries_llm, train_labels_llm_summary_20news,
                      test_labels_llm_summary_20news):
    """"
    Vectorizes LLM-generated summaries using TF-IDF, Doc2Vec, and BERT.
    """

    # 1) TF-IDF
    train_tfidf_sum, test_tfidf_sum, tfidf_vectorizer_sum = vectorize_tfidf(train_summaries_llm,
                                                                              test_summaries_llm, max_features=5000)

    # 2) Doc2Vec
    doc2vec_model_sum = train_doc2vec(train_summaries_llm)
    train_doc2vec_sum = vectorize_doc2vec(doc2vec_model_sum, train_summaries_llm)
    test_doc2vec_sum = vectorize_doc2vec(doc2vec_model_sum, test_summaries_llm)

    # 3) BERT (SentenceTransformer)
    train_bert_sum, test_bert_sum, bert_model_sum = vectorize_bert(train_summaries_llm, test_summaries_llm,
                                                                     model_name='all-MiniLM-L6-v2')

    #Save Everything
    sparse.save_npz('data/20news/train_summaries_tfidf_5000.npz', train_tfidf_sum)
    sparse.save_npz('data/20news/test_summaries_tfidf_5000.npz', test_tfidf_sum)
    with open('data/20news/tfidf_vectorizer_sum.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer_sum, f)
    np.save('data/20news/train_doc2vec_summaries_def.npy', train_doc2vec_sum)
    np.save('data/20news/test_doc2vec_summaries_def.npy', test_doc2vec_sum)
    doc2vec_model_sum.save('data/20news/doc2vec_model_summaries.model')
    np.save('data/20news/train_bert_summaries.npy', train_bert_sum)
    np.save('data/20news/test_bert_summaries.npy', test_bert_sum)
    bert_model_sum.save('data/20news/bert_sentence_transformer_model_llmsumary')
    np.save('data/labels/train_labels_summaries_llm.npy', train_labels_llm_summary_20news)
    np.save('data/labels/test_labels_summaries_llm.npy', test_labels_llm_summary_20news)


def vectorize_lsa_summary(train_summaries_lsa, test_summaries_lsa, train_labels_lsa_summary_20news,
                  test_labels_lsa_summary_20news):
    """
    Vectorizes LSA-generated summaries using TF-IDF, Doc2Vec, and BERT.
    """
    # TF-IDF
    train_tfidf_sum_lsa, test_tfidf_sum_lsa, tfidf_vectorizer_sum_lsa = vectorize_tfidf(train_summaries_lsa,
                                                                                        test_summaries_lsa,
                                                                                        max_features=5000)
    # Doc2Vec
    doc2vec_model_sum_lsa = train_doc2vec(train_summaries_lsa)
    train_doc2vec_sum_lsa = vectorize_doc2vec(doc2vec_model_sum_lsa, train_summaries_lsa)
    test_doc2vec_sum_lsa = vectorize_doc2vec(doc2vec_model_sum_lsa, test_summaries_lsa)


    # BERT
    train_bert_sum_lsa, test_bert_sum_lsa, bert_model_sum_lsa = vectorize_bert(train_summaries_lsa,
                                                                                 test_summaries_lsa,
                                                                                 model_name='all-MiniLM-L6-v2')

    #Save Everything
    sparse.save_npz('data/20news/train_summaries_tfidf_lsa.npz', train_tfidf_sum_lsa)
    sparse.save_npz('data/20news/test_summaries_tfidf_lsa.npz', test_tfidf_sum_lsa)
    with open('data/20news/tfidf_vectorizer_sum_lsa.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer_sum_lsa, f)
    np.save('data/20news/train_doc2vec_summaries_lsa.npy', train_doc2vec_sum_lsa)
    np.save('data/20news/test_doc2vec_summaries_lsa.npy', test_doc2vec_sum_lsa)
    doc2vec_model_sum_lsa.save('data/20news/doc2vec_model_summaries_lsa.model')
    np.save('data/20news/train_bert_summaries_lsa.npy', train_bert_sum_lsa)
    np.save('data/20news/test_bert_summaries_lsa.npy', test_bert_sum_lsa)
    bert_model_sum_lsa.save('data/20news/bert_sentence_transformer_model_lsasumary')
    np.save('data/labels/train_labels_summaries_lsa.npy', train_labels_lsa_summary_20news)
    np.save('data/labels/test_labels_summaries_lsa.npy', test_labels_lsa_summary_20news)


def vectorize_agnews(df_agnews, summary_df_llm, summary_df_lsa):
    """
    Vectorizes the AGNews dataset and its summaries using pre-trained models.
    """
    # Load trained model and save all vectorizers
    tfidf_vectorizer = joblib.load("data/20news/tfidf_vectorizer_full.pkl")
    tfidf_agnews_full = tfidf_vectorizer.transform(df_agnews['text'].tolist())
    sparse.save_npz('data/agnews/tfidf_agnews_flull.npz', tfidf_agnews_full)

    doc2vec_model = Doc2Vec.load("data/20news/doc2vec_model_full.model")
    doc2vec_agnews_full = vectorize_doc2vec(doc2vec_model, df_agnews['text'].tolist())
    np.save('data/agnews/doc2vec_agnews_full.npy', np.array(doc2vec_agnews_full))

    bert_model = SentenceTransformer('data/20news/bert_sentence_transformer_model_full')
    bert_agnews_full = bert_model.encode(df_agnews['text'].tolist())
    np.save('data/agnews/bert_agnews_full.npy', bert_agnews_full)

    tfidf_vectorizer_sum = joblib.load("data/20news/tfidf_vectorizer_sum.pkl")
    tfidf_agnews_llm = tfidf_vectorizer_sum.transform(summary_df_llm['summary'].tolist())
    sparse.save_npz('data/agnews/tfidf_agnews_llm.npz', tfidf_agnews_llm)

    doc2vec_model_sum = Doc2Vec.load("data/20news/doc2vec_model_summaries.model")
    doc2vec_agnews_llm = vectorize_doc2vec(doc2vec_model_sum, summary_df_llm['summary'].tolist())
    np.save('data/agnews/doc2vec_agnews_llm.npy', np.array(doc2vec_agnews_llm))

    bert_model_sum = SentenceTransformer('data/20news/bert_sentence_transformer_model_llmsumary')
    bert_agnews_llm = bert_model_sum.encode(summary_df_llm['summary'].tolist())
    np.save('data/agnews/bert_agnews_llm.npy', bert_agnews_llm)

    tfidf_vectorizer_sum_lsa = joblib.load("data/20news/tfidf_vectorizer_sum_lsa.pkl")
    tfidf_agnews_lsa = tfidf_vectorizer_sum_lsa.transform(summary_df_lsa['summary'].tolist())
    sparse.save_npz('data/agnews/tfidf_agnews_lsa.npz', tfidf_agnews_lsa)

    doc2vec_model_sum_lsa = Doc2Vec.load("data/20news/doc2vec_model_summaries_lsa.model")
    doc2vec_agnews_lsa = vectorize_doc2vec(doc2vec_model_sum_lsa, summary_df_lsa['summary'].tolist())
    np.save('data/agnews/doc2vec_agnews_lsa.npy', np.array(doc2vec_agnews_lsa))

    bert_model_sum_lsa = SentenceTransformer('data/20news/bert_sentence_transformer_model_lsasumary')
    bert_agnews_lsa = bert_model_sum_lsa.encode(summary_df_lsa['summary'].tolist())
    np.save('data/agnews/bert_agnews_lsa.npy', bert_agnews_lsa)

    np.save('data/agnews/labels_agnews_full.npy', df_agnews['label'].values)
    np.save('data/agnews/labels_summaries_lsa_agnews.npy', summary_df_lsa['label'].values)
    np.save('data/agnews/labels_summaries_llm_agnews.npy', summary_df_llm['label'].values)