import os
import sys
import pandas as pd
import numpy as np
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LOGS_FOLDER, LOSS_LOGS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE


class TrainingManager:
    
    def __init__(self) -> None:
        args = self.parse_arguments()

        self.layers = args.layer
        self.activation = args.activation
        self.weight_initializer = args.weight_initializer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.check_args()

        self.training_dataset = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")    
        self.X, self.Y = self.standardize_data()

        self.initialize_model()
        self.train()


    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="Multilayer perceptron model training program",
                                         description="Training Manager Configuration",
                                         epilog="")
        parser.add_argument(
            "--layer",
            type=int,
            nargs="+",
            help="List of integers specifying the size of each layer, e.g., --layer 24 24 24.",
            default=LAYER
        )
        parser.add_argument(
            "--epochs", 
            type=int, 
            help="Number of epochs for training, e.g., --epochs 84.",
            default=EPOCHS
        )
        parser.add_argument(
            "--loss", 
            type=str, 
            choices=["binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"], 
            help="Loss function to use, e.g., --loss binaryCrossentropy. (choices available: \"binaryCrossentropy\", \"categoricalCrossentropy\", \"meanSquaredError\")",
            default=LOSS
        )
        parser.add_argument(
            "--batch_size", 
            type=int, 
            help="Batch size for training, e.g., --batch_size 8.",
            default=BATCH_SIZE
        )
        parser.add_argument(
            "--learning_rate", 
            type=float, 
            help="Learning rate for the optimizer, e.g., --learning_rate 0.0314.",
            default=LEARNING_RATE
        )        
        parser.add_argument(
            "--config_file", 
            type=str, 
            help="Path to a configuration file that contains the training settings."
        )
        args = parser.parse_args()

        if len(args.layer) > 1:
            args.activation = ["sigmoid"] * (len(args.layer) - 1) + ["softmax"]
            args.weight_initializer = ["heUniform"] * (len(args.layer) - 1)
        else:
            args.activation = ["sigmoid"]

        if args.config_file:
            if len(sys.argv) == 2:
                return self.parse_config_file(args)
            else:
                raise ValueError("Can't have a config_file and other arguments at the same time.")
        else:
            return args


    def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
        pass


    def check_args(self) -> None:
        if not all(isinstance(layer, int) and layer > 0 for layer in self.layers):
            raise ValueError("All layers must be positive integers. Example: --layer 24 48 24.")

        if not all(act in ["sigmoid", "softmax", "relu", "tanh"] for act in self.activation):
            raise ValueError(f"Invalid activation function detected. Allowed values are: sigmoid, softmax, relu, tanh. Provided: {self.activation}.")

        if not all(init in ["heUniform", "xavierUniform", "randomNormal"] for init in self.weight_initializer):
            raise ValueError(f"Invalid weight initializer detected. Allowed values are: heUniform, xavierUniform, randomNormal. Provided: {self.weight_initializer}.")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Epochs must be a positive integer. Example: --epochs 50.")

        if self.loss_function not in ["binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"]:
            raise ValueError(f"Invalid loss function: {self.loss_function}. Allowed values are: binaryCrossentropy, categoricalCrossentropy, meanSquaredError.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer. Example: --batch_size 16.")

        if not isinstance(self.learning_rate, (float, int)) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number. Example: --learning_rate 0.01.")


    def load_data(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        try:
            return(pd.read_csv(file_path))
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{file_path}' is empty.")
        except Exception as e:
            raise Exception(f"An error occured while reading the file '{file_path}': {e}")
        
    
    def standardize_data(self) -> None:
        features = self.training_dataset.drop(columns=['id', 'diagnosis'])
        labels = self.training_dataset['diagnosis']

        mean, std = features.mean(), features.std()
        X = (features - mean) / std
        Y = labels.map({'M': 1, 'B': 0})

        return X.to_numpy(), Y.to_numpy().reshape(-1, 1)


    def initialize_model(self) -> None:
        self.model = {}
        input_size = self.X.shape[1]

        for i, layer_size in enumerate(self.layers):
            self.model[f"W{i+1}"] = np.random.randn(input_size, layer_size) * 0.01
            self.model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size


    def forward_propagation(self) -> np.ndarray:
        activations = self.X
        for i in range(1, len(self.layers) + 1):
            weights = self.model[f"W{i}"]
            bias = self.model[f"b{i}"]
            layer_inputs = np.dot(activations, weights) + bias
            activations = self.sigmoid(layer_inputs)
        return activations
    

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))


    def mean_squared_error(self, predictions: np.ndarray) -> float:
        if predictions.shape != self.Y.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs true labels {self.Y.shape}")
        
        mse = np.mean((self.Y - predictions) ** 2)
        return mse


    def train(self) -> None:
        for epoch in range(self.epochs):
            predictions = self.forward_propagation()
            mse = self.mean_squared_error(predictions)
