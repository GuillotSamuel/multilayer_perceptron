import os
import sys
import re
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import CONFIGURATION_FILE, CONFIGURATION_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE


class Initialization:

    def initialize_parameters(training_data, validation_data,
                            training_results, validation_results) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Training Manager Configuration")

        parser.add_argument("--layer", type=int, nargs="+", default=LAYER, help="Size of each layer.")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
        parser.add_argument("--loss", type=str, choices=["binaryCrossentropy"], default=LOSS, help="Loss function.")
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")

        if len(sys.argv) > 1:
            args = parser.parse_args()
            args.num_inputs = count_inputs(training_data, validation_data)
            args.num_outputs = count_outputs(training_results, validation_results)
            args.layer = [args.num_inputs] + args.layer + [args.num_outputs]

            args.activation = ["sigmoid"] * (len(args.layer) - 1)
            args.activation[-1] = "softmax"
            args.weight_initializer = ["he_uniform"] * (len(args.layer) - 2) + ["he_uniform"]

            return args
        else:
            return Initialization.parse_config_file(training_data, validation_data,
                                                    training_results, validation_results)


    @staticmethod
    def parse_config_file(training_data, validation_data,
                          training_results, validation_results) -> argparse.Namespace:
        """ """
        args = argparse.Namespace()

        args.num_inputs = count_inputs(training_data, validation_data)
        args.num_outputs = count_outputs(training_results, validation_results)

        config_file_path = os.path.join(CONFIGURATION_PATH, CONFIGURATION_FILE)

        try:
            with open(config_file_path, "r") as f:
                config_content = f.read()

            layers_info = []
            activation_funcs = []
            weight_initializers = []

            layer_pattern = r"layers\.DenseLayer\(([^)]+)\)"
            layer_matches = re.findall(layer_pattern, config_content)

            for layer_args in layer_matches:
                if "input_shape" in layer_args:
                    neuron_count = args.num_inputs
                elif "output_shape" in layer_args:
                    neuron_count = args.num_outputs
                else:
                    neuron_match = re.search(r"^(\d+)", layer_args)
                    if neuron_match:
                        neuron_count = int(neuron_match.group(1))
                    else:
                        raise ValueError(f"Could not parse neuron count from: {layer_args}")
                
                layers_info.append(neuron_count)
                
                activation_match = re.search(r"activation=[\'\"]([^\'\"]+)", layer_args)
                if activation_match:
                    activation_funcs.append(activation_match.group(1))
                else:
                    activation_funcs.append("linear")
                
                initializer_match = re.search(r"weights_initializer=[\'\"]([^\'\"]+)", layer_args)
                if initializer_match:
                    initializer = initializer_match.group(1)
                    initializer = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', initializer).lower()
                    weight_initializers.append(initializer)
                else:
                    weight_initializers.append("he_uniform")
            
            args.layer = layers_info
            
            args.activation = activation_funcs[1:]
            
            args.weight_initializer = weight_initializers[1:]
            
            lr_match = re.search(r"learning_rate=([0-9.]+)", config_content)
            args.learning_rate = float(lr_match.group(1)) if lr_match else LEARNING_RATE
            
            batch_match = re.search(r"batch_size=(\d+)", config_content)
            args.batch_size = int(batch_match.group(1)) if batch_match else BATCH_SIZE
            
            epochs_match = re.search(r"epochs=(\d+)", config_content)
            args.epochs = int(epochs_match.group(1)) if epochs_match else EPOCHS
            
            loss_match = re.search(r"loss=[\'\"]([^\'\"]+)", config_content)
            args.loss = loss_match.group(1) if loss_match else LOSS
            
        except Exception as e:
            print(f"Error parsing config file: {e}")
            hidden_layers = LAYER
            args.layer = [args.num_inputs] + hidden_layers + [args.num_outputs]
            args.activation = ["sigmoid"] * (len(args.layer) - 1)
            args.activation[-1] = "softmax"
            args.weight_initializer = ["he_uniform"] * (len(args.layer) - 1)
            args.epochs = EPOCHS
            args.loss = LOSS
            args.batch_size = BATCH_SIZE
            args.learning_rate = LEARNING_RATE
        
        return args


    @staticmethod
    def initialize_weights(parameters, layer_dims, weight_initializer):
        """
        Initialize weights for neural network with different methods
        """
        L = len(layer_dims)

        for l in range(1, L):
            if weight_initializer[l - 1] == "he":
                # He normal init:
                # → Generates random values from a normal (bell-shaped) distribution
                # → Values are a bit larger than Xavier, scaled for ReLU activations
                # → Good when you use ReLU because it keeps the signal strong enough
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
            elif weight_initializer[l - 1] == "he_uniform":
                # He uniform init:
                # → Generates random values from a uniform distribution (flat between -limit and +limit)
                # → Values spread evenly in a small range, also adapted for ReLU
                limit = np.sqrt(6. / layer_dims[l-1])
                parameters[f'W{l}'] = np.random.uniform(-limit, limit, (layer_dims[l], layer_dims[l-1]))
            elif weight_initializer[l - 1] == "xavier":
                # Xavier init:
                # → Generates random values from a normal distribution (around 0)
                # → Values are smaller than He init, balanced for sigmoid/tanh
                # → Helps avoid saturation where neurons stop learning
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1./layer_dims[l-1])
            elif weight_initializer[l - 1] == "zero":
                # Zero init:
                # → All weights are set to 0
                # → BAD for training: all neurons behave the same, no learning happens
                # → Only useful for testing/debugging
                parameters[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
            else:
                # Small random init (default/old method):
                # → Generates very small random numbers close to 0
                # → Works for tiny networks, but not good for deep ones
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01

            # Biases are always initialized to 0
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
