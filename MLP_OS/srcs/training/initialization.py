import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE


class Initialization:
    
    def parse_arguments(self) -> argparse.Namespace:
       pass
    
    
    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Training Manager Configuration")
        parser.add_argument("--layer", type=int, nargs="+", default=LAYER, help="Size of each layer.")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
        parser.add_argument("--loss", type=str, choices=["binaryCrossentropy"], default=LOSS, help="Loss function.")
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")
        args = parser.parse_args()

        args.layer.append(self.num_outputs)

        if len(args.layer) > 1:
            args.activation = ["sigmoid"] * (len(args.layer) - 1) + ["softmax"]
            args.weight_initializer = ["heUniform"] * (len(args.layer) - 1) + ["heUniform"]
        else:
            args.activation = ["sigmoid"]
            args.weight_initializer = ["heUniform"]

    
    def parse_config_file(self) -> argparse.Namespace:
        pass
    
    
    def check_arguments(self) -> None:
        pass

    
    @staticmethod
    def count_outputs(self) -> int:
        training_results = self.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")
        validation_results = self.load_file(f"{PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}")
        
        all_results = pd.concat([training_results, validation_results])

        num_outputs = all_results['diagnosis'].nunique()
        
        return num_outputs
    
    
    @staticmethod
    def initialize_weights(layer_dims, initialization="he"):
        """
        Initialize weights for neural network with different methods
        """
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            if initialization == "he":
                # He initialization for ReLU
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
            elif initialization == "he_uniform":
                # He Uniform initialization
                limit = np.sqrt(6. / layer_dims[l-1])
                parameters[f'W{l}'] = np.random.uniform(-limit, limit, (layer_dims[l], layer_dims[l-1]))
            elif initialization == "xavier":
                # Xavier initialization for tanh
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1./layer_dims[l-1])
            elif initialization == "zero":
                parameters[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
            else:  # small random
                parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        return parameters


    @staticmethod
    def one_hot_encode(y, num_classes=None):
        """
        Convert class labels to one-hot encoded format
        """
        if num_classes is None:
            num_classes = len(np.unique(y))
        return np.eye(num_classes)[y.reshape(-1)]
