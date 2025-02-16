import numpy as np

class Activation:
    
    def activation_g(Z, activation_type, derivative=False):
        """ Selects the activation function based on the input string """
        if activation_type == "sigmoid":
            A = Activation.sigmoid(Z, derivative)  # Commonly used for binary classification, smooth activation
        elif activation_type == "tanh":
            A = Activation.tanh(Z, derivative)  # Similar to sigmoid but centered around zero, avoids bias shift
        elif activation_type == "relu":
            A = Activation.relu(Z, derivative)  # Most popular activation, solves vanishing gradient for positive values
        elif activation_type == "leakyRelu":
            A = Activation.leaky_relu(Z, derivative)  # Variation of ReLU, allows small gradients for negative values
        elif activation_type == "elu":
            A = Activation.elu(Z, derivative)  # Similar to Leaky ReLU but smoother transition for negative values
        elif activation_type == "softmax":
            A = Activation.softmax(Z)  # Used in multi-class classification, converts logits to probabilities
        elif activation_type == "swish":
            A = Activation.swish(Z, derivative)  # Self-gated activation, improves performance in deep networks
        elif activation_type == "gelu":
            A = Activation.gelu(Z, derivative)  # Smooth approximation of ReLU, used in transformer models
        elif activation_type == "selu":
            A = Activation.selu(Z, derivative)  # Self-normalizing activation, helps with stable training
        elif activation_type == "mish":
            A = Activation.mish(Z, derivative)  # Smooth and non-monotonic, alternative to Swish for better training
        else:
            raise ValueError("Activation function not supported")  # Handles invalid activation function input
        return A


    def sigmoid(x, derivative=False):
        """
        Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        """
        sigmoid_x = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x
    
    
    def tanh(x, derivative=False):
        """
        Hyperbolic tangent activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        """
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)


    def relu(x, derivative=False):
        """
        Rectified Linear Unit (ReLU): f(x) = max(0, x)
        """
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)


    def leaky_relu(x, alpha=0.01, derivative=False):
        """
        Leaky ReLU: f(x) = max(αx, x) where α is a small positive constant
        """
        if derivative:
            return np.where(x > 0, 1, alpha)
        return np.where(x > 0, x, x * alpha)


    def elu(x, alpha=1.0, derivative=False):
        """
        Exponential Linear Unit (ELU): f(x) = x if x > 0 else α(e^x - 1)
        """
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def softmax(x, derivative=False):
        """
        Softmax activation function: f(x_i) = e^(x_i) / Σ(e^(x_j))
        Note: derivative not implemented as it requires Jacobian matrix
        """
        if derivative:
            raise NotImplementedError("Softmax derivative requires Jacobian matrix calculation")
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def swish(x, beta=1.0, derivative=False):
        """
        Swish activation function: f(x) = x * sigmoid(βx)
        """
        sigmoid_bx = 1 / (1 + np.exp(-beta * x))
        if derivative:
            return beta * sigmoid_bx + x * beta * sigmoid_bx * (1 - sigmoid_bx)
        return x * sigmoid_bx


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


    def selu(x, derivative=False):
        """
        Scaled Exponential Linear Unit (SELU)
        """
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        if derivative:
            return scale * np.where(x > 0, 1, alpha * np.exp(x))
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


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
