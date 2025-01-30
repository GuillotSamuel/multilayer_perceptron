import numpy as np


class Data_preprocessor:
    
    @staticmethod
    def one_hot_encode(y, num_classes=None):
        """
        Convert class labels to one-hot encoded format
        """
        if num_classes is None:
            num_classes = len(np.unique(y))
        return np.eye(num_classes)[y.reshape(-1)]


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
