from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from models.CustomTraining import CustomTraining
from models.CustomTesting import CustomTesting
from models.GridSearch import find_best_model

# Choose a representative training dataset (later replaced by real ones)
X_train_sample = []
y_train_sample = []

# Adjust acording to needs
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'learning_rate_init': [0.001, 0.01, 0.03],
    'max_iter': [500],
    'early_stopping': [True]
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

def training_process (tfidf_train_test_data_whole_5000, tfidf_train_test_data_llm_5000, tfidf_train_test_data_sumy_5000,
                        #tfidf_train_test_data_whole_1000, tfidf_train_test_data_llm_1000, tfidf_train_test_data_sumy_1000,
                        doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                        berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy):
    
    print ("find best svc")
    best_svc = find_best_model (SVC (probability=True), doc2vec_train_test_data_whole[2], doc2vec_train_test_data_whole[3], param_grid_svc)
    
    print ("find best mlp")
    best_mlp = find_best_model (MLPClassifier (random_state=42), doc2vec_train_test_data_whole[2], doc2vec_train_test_data_whole[3], param_grid_mlp)
    
    print ("find best rf")
    best_rf = find_best_model (RandomForestClassifier (random_state=42), doc2vec_train_test_data_whole[2], doc2vec_train_test_data_whole[3], param_grid_rf)

    classifiers = [best_svc, best_mlp, best_rf]

    train_test_data = [tfidf_train_test_data_whole_5000, tfidf_train_test_data_llm_5000, tfidf_train_test_data_sumy_5000,
                    #tfidf_train_test_data_whole_1000, tfidf_train_test_data_llm_1000, tfidf_train_test_data_sumy_1000,   
                    doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                    berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy]

    customTraining = CustomTraining (classifiers, train_test_data = train_test_data)
    customTraining.train ()
    customTraining.test ()
    customTraining.plot_results ()

def testing_process (tfidf_train_test_data_whole_5000, tfidf_train_test_data_llm_5000, tfidf_train_test_data_sumy_5000,
                        #tfidf_train_test_data_whole_1000, tfidf_train_test_data_llm_1000, tfidf_train_test_data_sumy_1000,
                        doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                        berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy, model_paths):
    
    train_test_data = [tfidf_train_test_data_whole_5000, tfidf_train_test_data_llm_5000, tfidf_train_test_data_sumy_5000,
                    #tfidf_train_test_data_whole_1000, tfidf_train_test_data_llm_1000, tfidf_train_test_data_sumy_1000,   
                    doc2vec_train_test_data_whole, doc2vec_train_test_data_llm, doc2vec_train_test_data_sumy,
                    berth_train_test_data_whole, berth_train_test_data_llm, berth_train_test_data_sumy]
    
    customTesting = CustomTesting (model_paths, train_test_data = train_test_data)
    customTesting.test_2 ()
    customTesting.plot_results ()