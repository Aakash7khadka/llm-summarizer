from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from models.CustomTraining import CustomTraining
from models.GridSearch import find_best_model

# Choose a representative training dataset (later replaced by real ones)
X_train_sample = []
y_train_sample = []

def training_process (this_is_a_dummy):
    
    best_svc = find_best_model (SVC (probability=True), X_train_sample, y_train_sample)
    best_mlp = find_best_model (MLPClassifier (random_state=42), X_train_sample, y_train_sample)
    best_rf = find_best_model (RandomForestClassifier (random_state=42),X_train_sample, y_train_sample)

    classifiers = [best_svc, best_mlp, best_rf]

    #tfidf_train_test_data = [X_vectors_train, y_vectors_train, X_vectors_test, y_vectors_test]
    tfidf_train_test_data_whole = [X_vectors_train_whole, y_vectors_train_whole, X_vectors_test_whole, y_vectors_test_whole]
    tfidf_train_test_data_llm = [X_vectors_train_llm, y_vectors_train_llm, X_vectors_test_llm, y_vectors_test_llm]
    tfidf_train_test_data_sumy = [X_vectors_train_sumy, y_vectors_train_sumy, X_vectors_test_sumy, y_vectors_test_sumy]

    #doc2vec_train_test_data = [X_vectors_train, y_vectors_train, X_vectors_test, y_vectors_test]
    doc2vec_train_test_data_whole = [X_vectors_train_whole, y_vectors_train_whole, X_vectors_test_whole, y_vectors_test_whole]
    doc2vec_train_test_data_llm = [X_vectors_train_llm, y_vectors_train_llm, X_vectors_test_llm, y_vectors_test_llm]
    doc2vec_train_test_data_sumy = [X_vectors_train_sumy, y_vectors_train_sumy, X_vectors_test_sumy, y_vectors_test_sumy]

    #berth_train_test_data = [X_vectors_train, y_vectors_train, X_vectors_test, y_vectors_test]
    berth_train_test_data_whole = [X_vectors_train_whole, y_vectors_train_whole, X_vectors_test_whole, y_vectors_test_whole]
    berth_train_test_data_llm = [X_vectors_train_llm, y_vectors_train_llm, X_vectors_test_llm, y_vectors_test_llm]
    berth_train_test_data_sumy = [X_vectors_train_sumy, y_vectors_train_sumy, X_vectors_test_sumy, y_vectors_test_sumy]

    train_test_data = [tfidf_train_test_data_whole, tfidf_train_test_data_llm, tfidf_train_test_data_sumy,
                    doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                    berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy]

    customTraining = CustomTraining (classifiers, train_test_data = train_test_data)
    customTraining.train ()
    customTraining.test ()
    customTraining.plot_results ()