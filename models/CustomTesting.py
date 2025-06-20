from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
from models.Run import Run
from collections import defaultdict
from models.trainers import save_model
import joblib

class CustomTesting:
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
        if self.classifiers == None: return
        for classifier in self.classifiers:
            print (classifier)

            if "berth" in classifier and "llm" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[7] [0],
                    self.train_test_data[7] [1],
                    self.train_test_data[7] [2],
                    self.train_test_data[7] [3]
                )
                self.runs.append (run)

            if "berth" in classifier and "sumy" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[8] [0],
                    self.train_test_data[8] [1],
                    self.train_test_data[8] [2],
                    self.train_test_data[8] [3]
                )
                self.runs.append (run)

            if "berth" in classifier and "whole" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[6] [0],
                    self.train_test_data[6] [1],
                    self.train_test_data[6] [2],
                    self.train_test_data[6] [3]
                )
                self.runs.append (run)

            if "doc2vec" in classifier and "llm" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[4] [0],
                    self.train_test_data[4] [1],
                    self.train_test_data[4] [2],
                    self.train_test_data[4] [3]
                )
                self.runs.append (run)

            if "doc2vec" in classifier and "sumy" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[5] [0],
                    self.train_test_data[5] [1],
                    self.train_test_data[5] [2],
                    self.train_test_data[5] [3]
                )
                self.runs.append (run)

            if "doc2vec" in classifier and "whole" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[3] [0],
                    self.train_test_data[3] [1],
                    self.train_test_data[3] [2],
                    self.train_test_data[3] [3]
                )
                self.runs.append (run)

            if "tfidf" in classifier and "llm" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[1] [0],
                    self.train_test_data[1] [1],
                    self.train_test_data[1] [2],
                    self.train_test_data[1] [3]
                )
                self.runs.append (run)

            if "tfidf" in classifier and "sumy" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[2] [0],
                    self.train_test_data[2] [1],
                    self.train_test_data[2] [2],
                    self.train_test_data[2] [3]
                )
                self.runs.append (run)

            if "tfidf" in classifier and "whole" in classifier:
                run = Run (
                    joblib.load(classifier),
                    self.train_test_data[0] [0],
                    self.train_test_data[0] [1],
                    self.train_test_data[0] [2],
                    self.train_test_data[0] [3]
                )
                self.runs.append (run)

        '''
        for d in self.train_test_data:
            print (d[2])
            print (d[3])
            print ()
        '''

    def test_2(self, debug=True):
        '''
        @abstract: Test the classifiers on the given vector representations and calculate accuracy, precision, recall, and F1.
        '''
        # Mapping from 20NG label to AG label
        map_20ng_to_ag = {
            0: 0,  # alt.atheism -> World
            1: 3,  # comp.graphics -> Sci/Tech
            2: 3,  # comp.os.ms-windows.misc -> Sci/Tech
            3: 3,  # comp.sys.ibm.pc.hardware -> Sci/Tech
            4: 3,  # comp.sys.mac.hardware -> Sci/Tech
            5: 3,  # comp.windows.x -> Sci/Tech
            6: 2,  # misc.forsale -> Business
            7: 2,  # rec.autos -> Business
            8: 2,  # rec.motorcycles -> Business
            9: 1,  # rec.sport.baseball -> Sports
            10: 1, # rec.sport.hockey -> Sports
            11: 3, # sci.crypt -> Sci/Tech
            12: 3, # sci.electronics -> Sci/Tech
            13: 3, # sci.med -> Sci/Tech
            14: 3, # sci.space -> Sci/Tech
            15: 0, # soc.religion.christian -> World
            16: 0, # talk.politics.guns -> World
            17: 0, # talk.politics.mideast -> World
            18: 0, # talk.politics.misc -> World
            19: 0, # talk.religion.misc -> World
        }

        for run in self.runs:
            classifier_name = run.classifier.__class__.__name__
            y_pred_20ng = run.classifier.predict(run.X_test)

            y_pred_ag = [map_20ng_to_ag.get(label, -1) for label in y_pred_20ng]

            run.metrics['accuracy'] = accuracy_score(run.y_test, y_pred_ag)
            run.metrics['precision'] = precision_score(run.y_test, y_pred_ag, average='weighted', zero_division=0)
            run.metrics['recall'] = recall_score(run.y_test, y_pred_ag, average='weighted', zero_division=0)
            run.metrics['f1'] = f1_score(run.y_test, y_pred_ag, average='weighted', zero_division=0)

            if debug:
                print(f"Classifier: {classifier_name}")
                for metric_name, value in run.metrics.items():
                    print(f"  {metric_name.capitalize()}: {value:.4f}")



    def test (self, debug = True):
        '''
        @abstract:  test the classifiers on the given vector representations and calculate accuracys, precisions, recalls and f1s
        '''
        for run in self.runs:
            classifier_name = run.classifier.__class__.__name__
            y_pred = run.classifier.predict (run.X_test)

            run.metrics['accuracy'] = accuracy_score (run.y_test, y_pred)
            run.metrics['precision'] = precision_score (run.y_test, y_pred, average='weighted', zero_division=0)
            run.metrics['recall'] = recall_score (run.y_test, y_pred, average='weighted', zero_division=0)
            run.metrics['f1'] = f1_score (run.y_test, y_pred, average='weighted', zero_division=0)

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
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f"{height:.4f}",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        rotation=90 
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