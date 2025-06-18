import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import load_npz

# from data.data_loader import load_cleaned_20news  # Or load_cleaned_agnews
from vectorization.vectorizers import vectorize_tfidf
from models.trainers import train_model
from evaluation.metrics import evaluate_model, plot_confusion_matrix
from models.training import training_process, testing_process

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def main():
    # Load and split data
    # logging.info("🚀 Starting model training pipeline...")

    # df = pd.read_csv("data/cleaned_20news.csv")  # Or load_cleaned_20news()
    # texts, labels = df['text'].tolist(), df['label'].tolist()

    # X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    #     texts, labels, test_size=0.2, random_state=42)

    # Vectorize with TF-IDF
    # X_train_vecs, vectorizer = vectorize_tfidf(X_train_texts)
    # X_test_vecs = vectorizer.transform(X_test_texts)

    '''

    X_berth_test_whole = np.load ("./vectorization/data/new/new/bert_test.npy")
    print (f"bert_test.npy : {X_berth_test_whole.shape}")
    X_berth_test_sumy = np.load ("./vectorization/data/new/new/test_bert_summaries_lsa.npy")
    print (f"test_bert_summaries_lsa.npy : {X_berth_test_sumy.shape}")
    X_berth_test_llm = np.load ("./vectorization/data/new/new/test_bert_summaries.npy")
    print (f"test_bert_summaries.npy : {X_berth_test_llm.shape}")

    print ("")

    X_d2v_test_whole = np.load ("./vectorization/data/new/new/doc2vec_test_def.npy")
    print (f"doc2vec_test_def.npy : {X_d2v_test_whole.shape}")
    X_d2v_test_sumy = np.load ("./vectorization/data/new/new/test_doc2vec_summaries_lsa.npy")
    print (f"test_doc2vec_summaries_lsa.npy : {X_d2v_test_sumy.shape}")
    X_d2v_test_llm = np.load ("./vectorization/data/new/new/test_doc2vec_summaries_def.npy")
    print (f"test_doc2vec_summaries_def.npy : {X_d2v_test_llm.shape}")

    print ("")

    X_tfidf_test_whole = load_npz ("./vectorization/data/new/new/tfidf_test_5000.npz")
    print (f"tfidf_test_5000.npy : {X_tfidf_test_whole.shape}")
    X_tfidf_test_sumy = load_npz ("./vectorization/data/new/new/test_summaries_tfidf_lsa.npz")
    print (f"test_summaries_tfidf_lsa.npz : {X_tfidf_test_sumy.shape}")
    X_tfidf_test_llm = load_npz ("./vectorization/data/new/new/test_summaries_tfidf_5000.npz")
    print (f"test_summaries_tfidf_5000.npz : {X_tfidf_test_llm.shape}")

    print ("")
    
    X_berth_train_whole = np.load ("./vectorization/data/new/new/bert_train.npy")
    print (f"bert_train.npy : {X_berth_train_whole.shape}")
    X_berth_train_sumy = np.load ("./vectorization/data/new/new/train_bert_summaries_lsa.npy")
    print (f"train_bert_summaries_lsa.npy : {X_berth_train_sumy.shape}")
    X_berth_train_llm = np.load ("./vectorization/data/new/new/train_bert_summaries.npy")
    print (f"train_bert_summaries.npy : {X_berth_train_llm.shape}")

    print ("")

    X_d2v_train_whole = np.load ("./vectorization/data/new/new/doc2vec_train_def.npy")
    print (f"doc2vec_train_def.npy : {X_d2v_train_whole.shape}")
    X_d2v_train_sumy = np.load ("./vectorization/data/new/new/train_doc2vec_summaries_lsa.npy")
    print (f"train_doc2vec_summaries_lsa.npy : {X_d2v_train_sumy.shape}")
    X_d2v_train_llm = np.load ("./vectorization/data/new/new/train_doc2vec_summaries_def.npy")
    print (f"train_doc2vec_summaries_def.npy : {X_d2v_train_llm.shape}")
    
    print ("")

    X_tfidf_train_whole = load_npz ("./vectorization/data/new/new/tfidf_train_5000.npz")
    print (f"tfidf_train_5000.npz : {X_tfidf_train_whole.shape}")
    X_tfidf_train_sumy = load_npz ("./vectorization/data/new/new/train_summaries_tfidf_lsa.npz")
    print (f"train_summaries_tfidf_lsa.npz : {X_tfidf_train_sumy.shape}")
    X_tfidf_train_llm = load_npz ("./vectorization/data/new/new/train_summaries_tfidf_5000.npz")
    print (f"train_summaries_tfidf_5000.npy : {X_tfidf_train_llm.shape}")

    print ("")

    y_test_whole = np.load ("./vectorization/data/new/new/test_labels_5000.npy")
    print (f"test_labels_5000.npy : {y_test_whole.shape}")
    y_test_sumy = np.load ("./vectorization/data/new/new/test_labels_summaries_lsa.npy")
    print (f"test_labels_summaries_lsa.npy : {y_test_sumy.shape}")
    y_test_llm = np.load ("./vectorization/data/new/new/test_labels_summaries_llm.npy")
    print (f"test_labels_summaries_llm.npy : {y_test_llm.shape}")
    
    print ("")

    y_train_whole = np.load ("./vectorization/data/new/new/train_labels_5000.npy")
    print (f"train_labels_5000.npy : {y_train_whole.shape}")
    y_train_sumy = np.load ("./vectorization/data/new/new/train_labels_summaries_lsa.npy")
    print (f"train_labels_summaries_lsa.npy : {y_train_sumy.shape}")
    y_train_llm = np.load ("./vectorization/data/new/new/train_labels_summaries_llm.npy")
    print (f"train_labels_summaries_llm.npy : {y_train_llm.shape}")


    


    n_train_labels = 14170
    n_test_labels = 3543
    
    tfidf_train_test_data_whole = [X_tfidf_train_whole, y_train_whole, X_tfidf_test_whole, y_test_whole]
    tfidf_train_test_data_llm = [X_tfidf_train_llm, y_train_llm, X_tfidf_test_llm, y_test_llm]
    tfidf_train_test_data_sumy = [X_tfidf_train_sumy, y_train_sumy, X_tfidf_test_sumy, y_test_sumy]
    #tfidf_train_test_data_sumy = [X_tfidf_train_sumy[:n_train_labels, :], y_train_sumy, X_tfidf_test_sumy[:n_test_labels, :], y_test_sumy]

    doc2vec_train_test_data_whole = [X_d2v_train_whole, y_train_whole, X_d2v_test_whole, y_test_whole]
    doc2vec_train_test_data_llm = [X_d2v_train_llm, y_train_llm, X_d2v_test_llm, y_test_llm]
    doc2vec_train_test_data_sumy = [X_d2v_train_sumy, y_train_sumy, X_d2v_test_sumy, y_test_sumy]

    berth_train_test_data_whole = [X_berth_train_whole, y_train_whole, X_berth_test_whole, y_test_whole]
    berth_train_test_data_llm = [X_berth_train_llm, y_train_llm, X_berth_test_llm, y_test_llm]
    berth_train_test_data_sumy = [X_berth_train_sumy, y_train_sumy, X_berth_test_sumy, y_test_sumy]

    training_process (tfidf_train_test_data_whole, tfidf_train_test_data_llm, tfidf_train_test_data_sumy,
                      doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                      berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy)
    
    return

    '''

    X_berth_ag_whole = np.load ("./vectorization/data/ag_news/ag_news/bert_agnews_full.npy")
    print (f"ag_bert.npy : {X_berth_ag_whole.shape}")
    X_berth_ag_sumy = np.load ("./vectorization/data/ag_news/ag_news/bert_agnews_lsa.npy")
    print (f"ag_bert_summaries_lsa.npy : {X_berth_ag_sumy.shape}")
    X_berth_ag_llm = np.load ("./vectorization/data/ag_news/ag_news/bert_agnews_llm.npy")
    print (f"ag_bert_summaries.npy : {X_berth_ag_llm.shape}")

    print ("")

    X_d2v_ag_whole = np.load ("./vectorization/data/ag_news/ag_news/doc2vec_agnews_full.npy")
    print (f"ag_doc2vec.npy : {X_d2v_ag_whole.shape}")
    X_d2v_ag_sumy = np.load ("./vectorization/data/ag_news/ag_news/doc2vec_agnews_lsa.npy")
    print (f"ag_doc2vec_summaries_lsa.npy : {X_d2v_ag_sumy.shape}")
    X_d2v_ag_llm = np.load ("./vectorization/data/ag_news/ag_news/doc2vec_agnews_llm.npy")
    print (f"ag_doc2vec_summaries_def.npy : {X_d2v_ag_llm.shape}")

    print ("")

    X_tfidf_ag_whole = load_npz ("./vectorization/data/ag_news/ag_news/tfidf_agnews_flull.npz")
    print (f"ag_tfidf.npy : {X_tfidf_ag_whole.shape}")
    X_tfidf_ag_sumy = load_npz ("./vectorization/data/ag_news/ag_news/tfidf_agnews_lsa.npz")
    print (f"ag_summaries_tfidf_lsa.npz : {X_tfidf_ag_sumy.shape}")
    X_tfidf_ag_llm = load_npz ("./vectorization/data/ag_news/ag_news/tfidf_agnews_llm.npz")
    print (f"ag_summaries_tfidf_5000.npz : {X_tfidf_ag_llm.shape}")

    print ("")

    y_ag_whole = np.load ("./vectorization/data/ag_news/ag_news/labels_agnews_full.npy")
    print (f"test_labels_5000.npy : {y_ag_whole.shape}")
    y_ag_sumy = np.load ("./vectorization/data/ag_news/ag_news/labels_summaries_lsa_agnews.npy")
    print (f"test_labels_summaries_lsa.npy : {y_ag_sumy.shape}")
    y_ag_llm = np.load ("./vectorization/data/ag_news/ag_news/labels_summaries_llm_agnews.npy")
    print (f"test_labels_summaries_llm.npy : {y_ag_llm.shape}")
    
    print ("")

    tfidf_train_test_data_whole = [None, None, X_tfidf_ag_whole, y_ag_whole]
    tfidf_train_test_data_llm = [None, None, X_tfidf_ag_llm, y_ag_llm]
    tfidf_train_test_data_sumy = [None, None, X_tfidf_ag_sumy, y_ag_sumy]
    
    doc2vec_train_test_data_whole = [None, None, X_d2v_ag_whole, y_ag_whole]
    doc2vec_train_test_data_llm = [None, None, X_d2v_ag_llm, y_ag_llm]
    doc2vec_train_test_data_sumy = [None, None, X_d2v_ag_sumy, y_ag_sumy]

    berth_train_test_data_whole = [None, None, X_berth_ag_whole, y_ag_whole]
    berth_train_test_data_llm = [None, None, X_berth_ag_llm, y_ag_llm]
    berth_train_test_data_sumy = [None, None, X_berth_ag_sumy, y_ag_sumy]

        
    folder_path = "./models/"
    joblib_files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and file.endswith(".joblib"):
            # Get relative path
            relative_path = os.path.relpath(full_path, folder_path)
            #if "svc" in relative_path:
            #    continue
            joblib_files.append(f"{folder_path}{relative_path}")

    testing_process (tfidf_train_test_data_whole, tfidf_train_test_data_llm, tfidf_train_test_data_sumy,
                      doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                      berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy, model_paths=joblib_files)
    

    return 

    #train and evaluate models
    #training_process (None)

    # Train and evaluate models
    model_types = ["svm", "mlp", "rf"]
    for model_type in model_types:
        logging.info(f"🔧 Training model: {model_type.upper()}")

        model = train_model(X_train_vecs, y_train, model_type=model_type)
        y_pred = model.predict(X_test_vecs)

        metrics = evaluate_model(y_test, y_pred)
        logging.info(f"📊 {model_type.upper()} metrics: {metrics}")

        # Optional: Save confusion matrix
        cm_path = f"outputs/confusion_matrix_{model_type}.png"
        plot_confusion_matrix(y_test, y_pred, labels=sorted(
            set(labels)), save_path=cm_path, show=False)


if __name__ == "__main__":
    main()
