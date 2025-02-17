import os
import sys
import argparse
import re
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE


class Initialization:
    
    def parsing(training_data, validation_data,
                training_results, validation_results) -> argparse.Namespace:
        """ Parse arguments from command line or configuration file """
        if len(sys.argv) > 1 and sys.argv[1].endswith(".config"):
            return Initialization.parse_config_file(sys.argv[1])
        else:
            return Initialization.parse_arguments(training_data, validation_data,
                                                   training_results, validation_results)


    def parse_arguments(training_data, validation_data,
                        training_results, validation_results) -> argparse.Namespace:
        """ Parse arguments from command line """
        parser = argparse.ArgumentParser(description="Training Manager Configuration")

        parser.add_argument("--layer", type=int, nargs="+", default=LAYER, help="Size of each layer.")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
        parser.add_argument("--loss", type=str, choices=["binaryCrossentropy", "categoricalCrossentropy"], default=LOSS, help="Loss function.")
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")

        args = parser.parse_args()
        
        args.num_inputs = count_inputs(training_data, validation_data)
        args.num_outputs = count_outputs(training_results, validation_results)
        
        args.layer = [args.num_inputs] + args.layer + [args.num_outputs]
        
        # Default model settings
        args.activation = ["relu"] * (len(args.layer) - 1)
        args.activation[-1] = "softmax"
        args.weight_initializer = ["he_uniform"] * (len(args.layer) - 2) + ["he_uniform"]

        return args


    def parse_config_file(config_path: str) -> argparse.Namespace:
        """ Parse configuration file for model parameters """
        layer_sizes = []
        activations = []
        weight_initializers = []
        loss = LOSS
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE
        epochs = EPOCHS

        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                
                # Dense layers parsing
                match = re.match(r".*DenseLayer\(([^)]+)\).*", line)
                if match:
                    params = match.group(1).split(",")
                    size = int(params[0].strip()) if params[0].strip().isdigit() else None
                    activation = None
                    weight_initializer = None

                    for param in params[1:]:
                        param = param.strip()
                        if "activation=" in param:
                            activation = param.split("=")[1].strip().strip("'")
                        if "weights_initializer=" in param:
                            weight_initializer = param.split("=")[1].strip().strip("'")

                    if size:
                        layer_sizes.append(size)
                    activations.append(activation or "relu")
                    weight_initializers.append(weight_initializer or "he_uniform")

                # Training parameters parsing
                match_train = re.search(r"loss='([^']+)', learning_rate=([\d\.]+), batch_size=(\d+), epochs=(\d+)", line)
                if match_train:
                    loss = match_train.group(1)
                    learning_rate = float(match_train.group(2))
                    batch_size = int(match_train.group(3))
                    epochs = int(match_train.group(4))

        args = argparse.Namespace()
        args.layer = layer_sizes
        args.activation = activations
        args.weight_initializer = weight_initializers
        args.loss = loss
        args.learning_rate = learning_rate
        args.batch_size = batch_size
        args.epochs = epochs

        return args


    @staticmethod
    def initialize_weights(parameters, layer_dims, weight_initializer):
        """
        Initialize weights for neural network with different methods
        """
        L = len(layer_dims)

        for l in range(1, L):
            if weight_initializer[l - 1] == "he":
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
            elif weight_initializer[l - 1] == "he_uniform":
                limit = np.sqrt(6. / layer_dims[l-1])
                parameters[f'W{l}'] = np.random.uniform(-limit, limit, (layer_dims[l], layer_dims[l-1]))
            elif weight_initializer[l - 1] == "xavier":
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1./layer_dims[l-1])
            elif weight_initializer[l - 1] == "zero":
                parameters[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
            else:
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01

            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        return parameters


def count_outputs(training_results, validation_results) -> int:
    """ Count number of outputs in the dataset """
    all_results = pd.concat([training_results, validation_results])

    num_outputs = all_results['label'].nunique()

    return num_outputs


def count_inputs(training_data, validation_data) -> int:
    """ Count number of inputs in the dataset """
    num_inputs = training_data.shape[1]

    if validation_data.shape[1] != num_inputs:
        raise ValueError("Training inputs don't match with Validation inputs.")
    
    return num_inputs
