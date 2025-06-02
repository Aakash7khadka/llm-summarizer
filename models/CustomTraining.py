from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
from models.Run import Run
from collections import defaultdict
from models.trainers import save_model

class CustomTraining:
    '''
    @abstract:  class that implements the training and testing logic
                for a list of classifiers on a list of vector representations.
                so for each vector representation we have a run for each classifier.
    '''
    def __init__ (self, classifiers: list, train_test_data: list[list]):
        '''
        @abstract:  initialize the custom training with the given parameters
        '''
        self.classifiers = classifiers
        self.train_test_data = train_test_data
        self.runs = []
        self._create_runs ()

    def _create_runs (self):
        '''
        @abstract:  create the runs for each classifier and vector representation
        '''
        for classifier in self.classifiers:
            for vector_representation in self.train_test_data:
                run = Run (
                    clone (classifier),
                    vector_representation[0],
                    vector_representation[1],
                    vector_representation[2],
                    vector_representation[3]
                )
                self.runs.append (run)

    def train (self, debug = True):
        '''
        @abstract:  train the classifiers on the given vector representations
        '''
        for run in self.runs:
            if debug: print (f"Training Classifier: {run.classifier}")
            run.classifier.fit (run.X_train, run.y_train)

        save_model (run.classifier)

    def test (self, debug = True):
        '''
        @abstract:  test the classifiers on the given vector representations and calculate accuracys, precisions, recalls and f1s
        '''
        for run in self.runs:
            classifier_name = run.classifier.__class__.__name__
            y_pred = run.classifier.predict (run.X_test)

            run.metrics['accuracy'] = accuracy_score (run.y_test, y_pred)
            run.metrics['precision'] = precision_score (run.y_test, y_pred, average='weighted', zero_division=0)
            run.metrics['recall'] = recall_score (run.y_test, y_pred, average='weighted')
            run.metrics['f1'] = f1_score (run.y_test, y_pred, average='weighted')

            if debug:
                print (f"Classifier: {classifier_name}")
                for metric_name, value in run.metrics.items ():
                    print (f"  {metric_name.capitalize()}: {value:.4f}")

    def plot_results (self):
        '''
        @abstract:  plot the results of the runs aggregated on classification method
        '''
        metrics_results = defaultdict (lambda: {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        })

        for run in self.runs:
            classifier_name = run.classifier.__class__.__name__
            for metric_name, value in run.metrics.items ():
                metrics_results[classifier_name][metric_name].append (value)

        for classifier_name, metrics in metrics_results.items ():
            metric_names = list (metrics.keys ())
            n_metrics = len (metric_names)
            n_runs = len (metrics[metric_names[0]])

            x = np.arange (n_runs)  # Vector representation indices
            bar_width = 0.2

            plt.figure (figsize=(10, 6))
            for i, metric_name in enumerate (metric_names):
                values = metrics[metric_name]
                positions = x + i * bar_width
                bars = plt.bar (positions, values, width=bar_width, label=metric_name.capitalize ())

                for bar in bars:
                    height = bar.get_height ()
                    plt.text (
                        bar.get_x () + bar.get_width () / 2,
                        height + 0.01,
                        f"{height:.4f}",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            plt.title (f"Performance Metrics per Run — {classifier_name}")
            plt.xlabel ("Vector Representation Index")
            plt.ylabel ("Score")
            plt.ylim (0.0, 1.1)
            plt.xticks (x + bar_width * (n_metrics - 1) / 2, [str (i) for i in range (n_runs)])
            plt.legend ()
            plt.grid (axis='y')
            plt.tight_layout ()
            plt.show ()