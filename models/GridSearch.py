from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

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


def find_best_model (model, X_train_sample, y_train_sample):
    grid_model = GridSearchCV (model, param_grid_svc, cv=5, n_jobs=-1, scoring='f1_weighted')
    grid_model.fit (X_train_sample, y_train_sample)
    best_model = grid_model.best_estimator_
    print ("Best Model:", grid_model.best_params_)
    return best_model


