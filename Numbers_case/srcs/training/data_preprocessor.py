import os
import sys
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
    def normalize_data(X, X_min, X_max, X_mean, X_std, method="minmax"):
        """
        Normalize input data using different methods
        """
        print("Normalization is starting...")
        if method == "minmax":
            return (X - X_min) / (X_max - X_min)
        elif method == "zscore":
            return (X - X_mean) / X_std
        elif method == "l2":
            return X / np.sqrt(np.sum(X**2, axis=0))
        elif method == "pixelNorm":
            return X / 255.0
        else:
            raise ValueError(f"Normalization method not supported : {method}")
