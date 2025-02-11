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
            cost = -np.mean(np.sum(Y * np.log(A + epsilon), axis=1))
        elif cost_function == "categoricalCrossentropy":
            cost = -np.mean(np.sum(Y * np.log(A + epsilon), axis=1))
        elif cost_function == "mse":
            cost = np.mean(np.square(A - Y))
        else:
            raise ValueError("Cost function not supported")
            
        return cost
