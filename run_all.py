import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import load_npz

# from data.data_loader import load_cleaned_20news  # Or load_cleaned_agnews
from vectorization.vectorizers import vectorize_tfidf
from models.trainers import train_model, save_model
from evaluation.metrics import evaluate_model, plot_confusion_matrix
from models.training import training_process, testing_process

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

import pickle
from scipy import sparse
import json
from vectorization.vectorize_save import (
    vectorize_full_dataset,
    vectorize_llm_summary,
    vectorize_lsa_summary,
    vectorize_agnews
)

from vectorization.preprocessing import prepare_dataframe
from llm.generate_summaries import generate_and_save_summaries_faster

LOAD_SUMMARIES = True
USE_LIGHT_VERSION = True

# from data.data_loader import load_cleaned_20news  # Or load_cleaned_agnews

from models.trainers import train_model, save_model
from evaluation.metrics import evaluate_model, plot_confusion_matrix

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def split_data(df, text_col='text', label_col='label', test_size=0.2, random_state=42):
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )


def main():
    # Load and split data
    logging.info("🚀 Starting model training pipeline...")

    df_20news = pd.read_csv("data/cleaned_20news_light.csv")  # Or load_cleaned_20news()
    # Load ag_news dataset
    df_agnews = pd.read_csv("data/cleaned_agnews_light.csv")
    
    # Change LOAD_SUMMARIES to False if you want to generate summaries
    # This creates summaries and saves them to JSON files inside the data folder
    if not LOAD_SUMMARIES:
        # Create summaries for agnews dataset
        generate_and_save_summaries_faster(
            input_csv="data/cleaned_agnews_light.csv",
            llm_json=f"data/summaries_agnews_llm.json",
            lsa_json=f"data/summaries_agnews_lsa.json")

        # Create summaries for 20news dataset
        generate_and_save_summaries_faster(
            input_csv="data/cleaned_20news_light.csv",
            llm_json=f"data/summaries_20news_llm.json",
            lsa_json=f"data/summaries_20news_lsa.json")


    # Load summaries from JSON files
    with open('data/summaries_20news_llm.json', 'r',
            encoding='utf-8') as f:  # Have to fifgure out how are we importing the summaries
        summaries_20news_llm_dict = json.load(f)

    with open('data/summaries_20news_lsa.json', 'r', encoding='utf-8') as f:
        summaries_20news_lsa_dict = json.load(f)


    # Load summaries from JSON files
    with open('data/summaries_agnews_llm.json', 'r', encoding='utf-8') as f:
        summaries_agnews_llm_dict = json.load(f)

    with open('data/summaries_agnews_lsa.json', 'r', encoding='utf-8') as f:
        summaries_agnews_lsa_dict = json.load(f)

    # Clean and align summaries with full dataset
    df_20news_cleaned, summary_df_llm, summary_df_lsa = prepare_dataframe(df_20news, summaries_20news_llm_dict,
                                                                          summaries_20news_lsa_dict)

    df_agnews_cleaned, summary_df_llm_agnews, summary_df_lsa_agnews = prepare_dataframe(df_agnews,
                                                                                        summaries_agnews_llm_dict,
                                                                                        summaries_agnews_lsa_dict)

    # Split full dataset
    train_texts, test_texts, train_labels, test_labels = split_data(df_20news_cleaned, 'text', 'label')

    # Split LLM summaries
    train_summaries_llm, test_summaries_llm, train_labels_llm, test_labels_llm = split_data(summary_df_llm, 'summary',
                                                                                            'label')
    # Split LSA summaries
    train_summaries_lsa, test_summaries_lsa, train_labels_lsa, test_labels_lsa = split_data(summary_df_lsa, 'summary',
                                                                                            'label')

    # Vectorzing and saving 20_news full text dataset
    vectorize_full_dataset(train_texts, test_texts, train_labels, test_labels)

    # Vectorizing and saving 20News LLM summaries
    vectorize_llm_summary(train_summaries_llm, test_summaries_llm,
                          train_labels_llm, test_labels_llm)

    # Vectorizing and saving 20News LSA summaries
    vectorize_lsa_summary(train_summaries_lsa, test_summaries_lsa,
                          train_labels_lsa, test_labels_lsa)

    # Vectorizing and saving AGNews dataset using pretrained 20News models
    vectorize_agnews(df_agnews_cleaned, summary_df_llm_agnews, summary_df_lsa_agnews)

    

    

    X_berth_test_whole = np.load ("data/20news/bert_test.npy")
    print (f"bert_test.npy : {X_berth_test_whole.shape}")
    X_berth_test_sumy = np.load ("data/20news/test_bert_summaries_lsa.npy")
    print (f"test_bert_summaries_lsa.npy : {X_berth_test_sumy.shape}")
    X_berth_test_llm = np.load ("data/20news/test_bert_summaries.npy")
    print (f"test_bert_summaries.npy : {X_berth_test_llm.shape}")

    print ("")

    X_d2v_test_whole = np.load ("data/20news/doc2vec_test_def.npy")
    print (f"doc2vec_test_def.npy : {X_d2v_test_whole.shape}")
    X_d2v_test_sumy = np.load ("data/20news/test_doc2vec_summaries_lsa.npy")
    print (f"test_doc2vec_summaries_lsa.npy : {X_d2v_test_sumy.shape}")
    X_d2v_test_llm = np.load ("data/20news/test_doc2vec_summaries_def.npy")
    print (f"test_doc2vec_summaries_def.npy : {X_d2v_test_llm.shape}")

    print ("")

    X_tfidf_test_whole = load_npz ("data/20news/tfidf_test_5000.npz")
    print (f"tfidf_test_5000.npy : {X_tfidf_test_whole.shape}")
    X_tfidf_test_sumy = load_npz ("data/20news/test_summaries_tfidf_lsa.npz")
    print (f"test_summaries_tfidf_lsa.npz : {X_tfidf_test_sumy.shape}")
    X_tfidf_test_llm = load_npz ("data/20news/test_summaries_tfidf_5000.npz")
    print (f"test_summaries_tfidf_5000.npz : {X_tfidf_test_llm.shape}")

    print ("")
    
    X_berth_train_whole = np.load ("data/20news/bert_train.npy")
    print (f"bert_train.npy : {X_berth_train_whole.shape}")
    X_berth_train_sumy = np.load ("data/20news/train_bert_summaries_lsa.npy")
    print (f"train_bert_summaries_lsa.npy : {X_berth_train_sumy.shape}")
    X_berth_train_llm = np.load ("data/20news/train_bert_summaries.npy")
    print (f"train_bert_summaries.npy : {X_berth_train_llm.shape}")

    print ("")

    X_d2v_train_whole = np.load ("data/20news/doc2vec_train_def.npy")
    print (f"doc2vec_train_def.npy : {X_d2v_train_whole.shape}")
    X_d2v_train_sumy = np.load ("data/20news/train_doc2vec_summaries_lsa.npy")
    print (f"train_doc2vec_summaries_lsa.npy : {X_d2v_train_sumy.shape}")
    X_d2v_train_llm = np.load ("data/20news/train_doc2vec_summaries_def.npy")
    print (f"train_doc2vec_summaries_def.npy : {X_d2v_train_llm.shape}")
    
    print ("")

    X_tfidf_train_whole = load_npz ("data/20news/tfidf_train_5000.npz")
    print (f"tfidf_train_5000.npz : {X_tfidf_train_whole.shape}")
    X_tfidf_train_sumy = load_npz ("data/20news/train_summaries_tfidf_lsa.npz")
    print (f"train_summaries_tfidf_lsa.npz : {X_tfidf_train_sumy.shape}")
    X_tfidf_train_llm = load_npz ("data/20news/train_summaries_tfidf_5000.npz")
    print (f"train_summaries_tfidf_5000.npy : {X_tfidf_train_llm.shape}")

    print ("")

    y_test_whole = np.load ("data/20news/test_labels_5000.npy")
    print (f"test_labels_5000.npy : {y_test_whole.shape}")
    y_test_sumy = np.load ("data/labels/test_labels_summaries_lsa.npy")
    print (f"test_labels_summaries_lsa.npy : {y_test_sumy.shape}")
    y_test_llm = np.load ("data/labels/test_labels_summaries_llm.npy")
    print (f"test_labels_summaries_llm.npy : {y_test_llm.shape}")
    
    print ("")

    y_train_whole = np.load ("data/20news/train_labels_5000.npy")
    print (f"train_labels_5000.npy : {y_train_whole.shape}")
    y_train_sumy = np.load ("data/labels/train_labels_summaries_lsa.npy")
    print (f"train_labels_summaries_lsa.npy : {y_train_sumy.shape}")
    y_train_llm = np.load ("data/labels/train_labels_summaries_llm.npy")
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
    
    

    

    #
    #
    #  Pipeline ends here.
    #  everything else is evaluation
    #
    #

    #
    # rename models
    #
    os.rename ("models/final_model_0.joblib", "models/svc_tfidf_whole.joblib")
    os.rename ("models/final_model_1.joblib", "models/svc_tfidf_llm.joblib")
    os.rename ("models/final_model_2.joblib", "models/svc_tfidf_sumy.joblib")
    os.rename ("models/final_model_3.joblib", "models/svc_doc2vec_whole.joblib")
    os.rename ("models/final_model_4.joblib", "models/svc_doc2vec_llm.joblib")
    os.rename ("models/final_model_5.joblib", "models/svc_doc2vec_sumy.joblib")
    os.rename ("models/final_model_6.joblib", "models/svc_berth_whole.joblib")
    os.rename ("models/final_model_7.joblib", "models/svc_berth_llm.joblib")
    os.rename ("models/final_model_8.joblib", "models/svc_berth_sumy.joblib")

    os.rename ("models/final_model_9.joblib", "models/mlp_tfidf_whole.joblib")
    os.rename ("models/final_model_10.joblib", "models/mlp_tfidf_llm.joblib")
    os.rename ("models/final_model_11.joblib", "models/mlp_tfidf_sumy.joblib")
    os.rename ("models/final_model_12.joblib", "models/mlp_doc2vec_whole.joblib")
    os.rename ("models/final_model_13.joblib", "models/mlp_doc2vec_llm.joblib")
    os.rename ("models/final_model_14.joblib", "models/mlp_doc2vec_sumy.joblib")
    os.rename ("models/final_model_15.joblib", "models/mlp_berth_whole.joblib")
    os.rename ("models/final_model_16.joblib", "models/mlp_berth_llm.joblib")
    os.rename ("models/final_model_17.joblib", "models/mlp_berth_sumy.joblib")

    os.rename ("models/final_model_18.joblib", "models/rf_tfidf_whole.joblib")
    os.rename ("models/final_model_19.joblib", "models/rf_tfidf_llm.joblib")
    os.rename ("models/final_model_20.joblib", "models/rf_tfidf_sumy.joblib")
    os.rename ("models/final_model_21.joblib", "models/rf_doc2vec_whole.joblib")
    os.rename ("models/final_model_22.joblib", "models/rf_doc2vec_llm.joblib")
    os.rename ("models/final_model_23.joblib", "models/rf_doc2vec_sumy.joblib")
    os.rename ("models/final_model_24.joblib", "models/rf_berth_whole.joblib")
    os.rename ("models/final_model_25.joblib", "models/rf_berth_llm.joblib")
    os.rename ("models/final_model_26.joblib", "models/rf_berth_sumy.joblib")

    

    X_berth_ag_whole = np.load ("data/agnews/bert_agnews_full.npy")
    print (f"ag_bert.npy : {X_berth_ag_whole.shape}")
    X_berth_ag_sumy = np.load ("data/agnews/bert_agnews_lsa.npy")
    print (f"ag_bert_summaries_lsa.npy : {X_berth_ag_sumy.shape}")
    X_berth_ag_llm = np.load ("data/agnews/bert_agnews_llm.npy")
    print (f"ag_bert_summaries.npy : {X_berth_ag_llm.shape}")

    print ("")

    X_d2v_ag_whole = np.load ("data/agnews/doc2vec_agnews_full.npy")
    print (f"ag_doc2vec.npy : {X_d2v_ag_whole.shape}")
    X_d2v_ag_sumy = np.load ("data/agnews/doc2vec_agnews_lsa.npy")
    print (f"ag_doc2vec_summaries_lsa.npy : {X_d2v_ag_sumy.shape}")
    X_d2v_ag_llm = np.load ("data/agnews/doc2vec_agnews_llm.npy")
    print (f"ag_doc2vec_summaries_def.npy : {X_d2v_ag_llm.shape}")

    print ("")

    X_tfidf_ag_whole = load_npz ("data/agnews/tfidf_agnews_flull.npz")
    print (f"ag_tfidf.npy : {X_tfidf_ag_whole.shape}")
    X_tfidf_ag_sumy = load_npz ("data/agnews/tfidf_agnews_lsa.npz")
    print (f"ag_summaries_tfidf_lsa.npz : {X_tfidf_ag_sumy.shape}")
    X_tfidf_ag_llm = load_npz ("data/agnews/tfidf_agnews_llm.npz")
    print (f"ag_summaries_tfidf_5000.npz : {X_tfidf_ag_llm.shape}")

    print ("")

    y_ag_whole = np.load ("data/agnews/labels_agnews_full.npy")
    print (f"test_labels_5000.npy : {y_ag_whole.shape}")
    y_ag_sumy = np.load ("data/agnews/labels_summaries_lsa_agnews.npy")
    print (f"test_labels_summaries_lsa.npy : {y_ag_sumy.shape}")
    y_ag_llm = np.load ("data/agnews/labels_summaries_llm_agnews.npy")
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
    
    if USE_LIGHT_VERSION:
        os.remove ( "models/svc_tfidf_whole.joblib")
        os.remove ( "models/svc_tfidf_llm.joblib")
        os.remove ( "models/svc_tfidf_sumy.joblib")
        os.remove ( "models/svc_doc2vec_whole.joblib")
        os.remove ( "models/svc_doc2vec_llm.joblib")
        os.remove ( "models/svc_doc2vec_sumy.joblib")
        os.remove ( "models/svc_berth_whole.joblib")
        os.remove ( "models/svc_berth_llm.joblib")
        os.remove ( "models/svc_berth_sumy.joblib")

        os.remove ( "models/mlp_tfidf_whole.joblib")
        os.remove ( "models/mlp_tfidf_llm.joblib")
        os.remove ( "models/mlp_tfidf_sumy.joblib")
        os.remove ( "models/mlp_doc2vec_whole.joblib")
        os.remove ( "models/mlp_doc2vec_llm.joblib")
        os.remove ( "models/mlp_doc2vec_sumy.joblib")
        os.remove ( "models/mlp_berth_whole.joblib")
        os.remove ( "models/mlp_berth_llm.joblib")
        os.remove ( "models/mlp_berth_sumy.joblib")

        os.remove ( "models/rf_tfidf_whole.joblib")
        os.remove ( "models/rf_tfidf_llm.joblib")
        os.remove ( "models/rf_tfidf_sumy.joblib")
        os.remove ( "models/rf_doc2vec_whole.joblib")
        os.remove ( "models/rf_doc2vec_llm.joblib")
        os.remove ( "models/rf_doc2vec_sumy.joblib")
        os.remove ( "models/rf_berth_whole.joblib")
        os.remove ( "models/rf_berth_llm.joblib")
        os.remove ( "models/rf_berth_sumy.joblib")
    
    return 


if __name__ == "__main__":
    main()
