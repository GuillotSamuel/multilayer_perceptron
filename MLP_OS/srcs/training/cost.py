import numpy as np

class Cost:

    @staticmethod
    def compute_cost(AL, Y, cost_function="binary_crossentropy"):
        """
        Compute the cost using different loss functions
        """
        m = Y.shape[1]
        epsilon = 1e-15
        
        if cost_function == "binary_crossentropy":
            cost = -np.mean(Y * np.log(AL + epsilon) + (1-Y) * np.log(1-AL + epsilon))
        elif cost_function == "categorical_crossentropy":
            cost = -np.mean(np.sum(Y * np.log(AL + epsilon), axis=0))
        elif cost_function == "mse":
            cost = np.mean(np.square(AL - Y))
        else:
            raise ValueError("Cost function not supported")
            
        return cost
