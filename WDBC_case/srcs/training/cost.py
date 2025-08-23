import numpy as np

class Cost:

    @staticmethod
    def compute_loss(A, Y, cost_function="binary_crossentropy"):
        """
        Compute the cost using different loss functions
        methods:
        - binary crossentropy : used for binary classification (e.g., is it a cat (1) or not (0))
        - categorical crossentropy : used for multi-class classification with one-hot encoded labels 
        (e.g., is it a cat [1, 0, 0], dog [0, 1, 0], or bird [0, 0, 1])
        - mse : measures the average squared difference between predicted and actual values; mainly used for regression tasks
        """
        m = Y.shape[1]
        epsilon = 1e-15

        if cost_function == "binaryCrossentropy":
            cost = -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        elif cost_function == "categoricalCrossentropy":
            cost = -np.mean(np.sum(Y * np.log(A + epsilon), axis=1))
        elif cost_function == "mse":
            cost = np.mean(np.square(A - Y))
        else:
            raise ValueError("Cost function not supported")

        return cost
