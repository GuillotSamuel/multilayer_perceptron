import numpy as np
import pandas as pd


class Data_preprocessor:
    
    @staticmethod
    def one_hot_encode(y, num_classes=None):
        """
        Convert class labels to one-hot encoded format
        """
        if isinstance(y, pd.DataFrame):
            y = y.values

        y = np.array(y).flatten()

        if y.dtype.kind in {'U', 'S', 'O'}:
            unique_classes, y = np.unique(y, return_inverse=True)

        if num_classes is None:
            num_classes = len(np.unique(y))

        return np.eye(num_classes, dtype=int)[y]


    @staticmethod
    def normalize_data(X, method="minmax"):
        """
        Normalize input data using different methods
        """
        if method == "minmax":
            return (X - X.min()) / (X.max() - X.min())
        elif method == "zscore":
            return (X - X.mean()) / X.std()
        elif method == "l2":
            return X / np.sqrt(np.sum(X**2, axis=0))
        return X
