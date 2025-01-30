import numpy as np

class Activation:
    
    @staticmethod
    def sigmoid(x, derivative=False):
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        """
        sigmoid_x = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x
    
    
    @staticmethod
    def tanh(x, derivative=False):
        """
        Hyperbolic tangent activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        """
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)


    @staticmethod
    def relu(x, derivative=False):
        """
        Rectified Linear Unit (ReLU): f(x) = max(0, x)
        """
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)


    @staticmethod
    def leaky_relu(x, alpha=0.01, derivative=False):
        """
        Leaky ReLU: f(x) = max(αx, x) where α is a small positive constant
        """
        if derivative:
            return np.where(x > 0, 1, alpha)
        return np.where(x > 0, x, x * alpha)


    @staticmethod
    def elu(x, alpha=1.0, derivative=False):
        """
        Exponential Linear Unit (ELU): f(x) = x if x > 0 else α(e^x - 1)
        """
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def softmax(x, derivative=False):
        """
        Softmax activation function: f(x_i) = e^(x_i) / Σ(e^(x_j))
        Note: derivative not implemented as it requires Jacobian matrix
        """
        if derivative:
            raise NotImplementedError("Softmax derivative requires Jacobian matrix calculation")
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    @staticmethod
    def swish(x, beta=1.0, derivative=False):
        """
        Swish activation function: f(x) = x * sigmoid(βx)
        """
        sigmoid_bx = 1 / (1 + np.exp(-beta * x))
        if derivative:
            return beta * sigmoid_bx + x * beta * sigmoid_bx * (1 - sigmoid_bx)
        return x * sigmoid_bx


    @staticmethod
    def gelu(x, derivative=False):
        """
        Gaussian Error Linear Unit (GELU): f(x) = x * Φ(x)
        where Φ(x) is the cumulative distribution function of the standard normal distribution
        """
        if derivative:
            cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)
            return cdf + x * pdf
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


    @staticmethod
    def selu(x, derivative=False):
        """
        Scaled Exponential Linear Unit (SELU)
        """
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        if derivative:
            return scale * np.where(x > 0, 1, alpha * np.exp(x))
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


    @staticmethod
    def mish(x, derivative=False):
        """
        Mish activation function: f(x) = x * tanh(softplus(x))
        where softplus(x) = ln(1 + e^x)
        """
        softplus = np.log1p(np.exp(x))
        tanh_softplus = np.tanh(softplus)
        
        if derivative:
            sech_squared = 1 - tanh_softplus ** 2
            sigmoid = 1 / (1 + np.exp(-x))
            return tanh_softplus + x * sech_squared * sigmoid
        return x * tanh_softplus
