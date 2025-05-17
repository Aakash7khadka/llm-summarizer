from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split input features and labels into train and test sets.

    Args:
        X: Feature matrix (NumPy array, pandas DataFrame, or list)
        y: Target labels (list, Series, or array)
        test_size (float): Proportion for test data
        random_state (int): Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
