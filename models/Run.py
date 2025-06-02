import matplotlib.pyplot as plt
from collections import defaultdict
import math

class Run:
    '''
    @abstract:  class that contains the data and results of a single run
                of a classifier on a single vector representation
    '''
    def __init__ (self, classifier, X_train, y_train, X_test, y_test):
        '''
        @abstract:  initialize the run with the given parameters
        '''
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None
        }

    def __str__ (self):
        '''
        @abstract:  return a string representation of the run
        '''
        clf_name = self.classifier.__class__.__name__
        metrics_str = ', '.join (f"{k.capitalize ()}: {v:.4f}" for k, v in self.metrics.items () if v is not None)
        return f"{clf_name} — {metrics_str}"