import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def evaluate_model(y_true, y_pred, average: str = 'weighted') -> dict:
    """
    Calculate and return evaluation metrics: accuracy, precision, recall, and F1-score.
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    logging.info(f"Evaluation results: {results}")
    return results


def plot_confusion_matrix(
    y_true, y_pred, labels: list,
    title: str = "Confusion Matrix",
    save_path: str = None,
    show: bool = True
):
    """
    Plot a labeled confusion matrix. Optionally saves the figure to disk.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar=False
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
