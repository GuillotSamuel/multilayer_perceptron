import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE


class Initialization:

    def parse_arguments(training_data, validation_data,
                        training_results, validation_results) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Training Manager Configuration")

        parser.add_argument("--layer", type=int, nargs="+", default=LAYER, help="Size of each layer.")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
        parser.add_argument("--loss", type=str, choices=["binaryCrossentropy"], default=LOSS, help="Loss function.")
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")

        args = parser.parse_args()
        
        args.num_inputs = count_inputs(training_data, validation_data)
        args.num_outputs = count_outputs(training_results, validation_results)
        
        args.layer = [args.num_inputs] + args.layer + [args.num_outputs]
        
        # Default model settings
        args.activation = ["sigmoid"] * (len(args.layer) - 1)
        args.activation[-1] = "softmax"
        args.weight_initializer = ["he_uniform"] * (len(args.layer) - 2) + ["he_uniform"]

        return args


    # def parse_config_file(self) -> argparse.Namespace:


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
    all_results = pd.concat([training_results, validation_results])

    num_outputs = all_results['diagnosis'].nunique()

    return num_outputs


def count_inputs(training_data, validation_data) -> int:
    num_inputs = training_data.shape[1]

    if validation_data.shape[1] != num_inputs:
        raise ValueError("Training inputs don't match with Validation inputs.")
    
    return num_inputs
