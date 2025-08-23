import numpy as np

class Cost:

    @staticmethod
    def compute_loss(A, Y, cost_function="binary_crossentropy"):
        """
        Compute the cost using different loss functions
        """
        m = Y.shape[1]
        epsilon = 1e-15
        
        if cost_function == "binaryCrossentropy":
            cost = -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))  # Used for binary classification, measures log loss
        elif cost_function == "categoricalCrossentropy":
            cost = -np.mean(np.sum(Y * np.log(A + epsilon), axis=1))  # Used for multi-class classification, generalizes binary cross-entropy
        elif cost_function == "mse":
            cost = np.mean(np.square(A - Y))  # Mean Squared Error, common for regression tasks
        else:
            raise ValueError("Cost function not supported")

            
        return cost
