from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier




def find_best_model (model, X_train_sample, y_train_sample, param_grid):
    grid_model = GridSearchCV (model, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
    grid_model.fit (X_train_sample, y_train_sample)
    best_model = grid_model.best_estimator_
    print ("Best Model:", grid_model.best_params_)
    return best_model


