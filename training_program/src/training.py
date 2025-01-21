import os
import sys
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE

class TrainingManager:
    
    def __init__(self) -> None:
        """
        Constructor.

        Args:
            None
        Returns:
            None
        """
        args = self.parse_arguments()

        self.layers = args.layer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        
        self.check_args()

        self.x_training_dataset = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        self.standardize_data()


    def parse_arguments(self) -> argparse.Namespace:
        """
        Parses command-line arguments for training configuration.

        Returns:
            argparse.Namespace: Parsed arguments including:
                - `--layer` (list[int]): Sizes of each layer (default: LAYERS).
                - `--epochs` (int): Number of training epochs (default: EPOCHS).
                - `--loss` (str): Loss function (choices: "binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"; default: LOSS).
                - `--batch_size` (int): Batch size (default: BATCH_SIZE).
                - `--learning_rate` (float): Learning rate (default: LEARNING_RATE).

        Example:
            python train.py --config_file config.txt
            python train.py --layer 32 64 32 --epochs 50 --loss binaryCrossentropy --batch_size 16 --learning_rate 0.01
        """
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

        if args.config_file:
            if len(sys.argv) == 2:
                return self.parse_config_file(args)
            else:
                raise ValueError("")
        else:
            return args


    def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
        """
        Parses the configuration file and returns the parsed arguments.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            argparse.Namespace: Parsed arguments from the config file.
        """
        pass


    def check_args(self) -> None:
        """
        Check the validity of the arguments provided.

        Raises:
            ValueError: If any of the arguments is invalid.
        """
        if not all(isinstance(layer, int) and layer > 0 for layer in self.layers):
            raise ValueError("All layers must be positive integers. Example: --layer 24 48 24.")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("Epochs must be a positive integer. Example: --epochs 50.")

        if self.loss_function not in ["binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"]:
            raise ValueError(f"Invalid loss function: {self.loss_function}. Allowed values are: binaryCrossentropy, categoricalCrossentropy, meanSquaredError.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer. Example: --batch_size 16.")

        if not isinstance(self.learning_rate, (float, int)) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number. Example: --learning_rate 0.01.")


    def load_data(self, file_path: str) -> None:
        """
        Load data from a CSV file into a DataFrame.

        Args:
            None
        Returns:
            None
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        try:
            return(pd.read_csv(file_path))
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{file_path}' is empty.")
        except Exception as e:
            raise Exception(f"An error occured while reading the file '{file_path}': {e}")


    def standardize_data(self) -> None:
        """
        Standardize the data.
        
        Args:
            None
        Returns:
            None
        """
        training_features = self.x_training_dataset.drop(columns=['id', 'diagnosis'])
        training_labels_diag = self.x_training_dataset['diagnosis']

        mean = training_features.mean()
        std = training_features.std()
        standardized_features = (training_features - mean) / std
        self.x_training_dataset = standardized_features
        self.y_training_dataset = training_labels_diag


    def stochastic_gradient_descent(self):
        """
        Apply backpropagation on all values to update weights and biases.
        
        Args:
            None
        Returns:
            None
        """
        pass
        
    def backprop(self):
        """
        
        """
        pass
            
    def evaluate(self) -> int:
        """
        
        """
        pass


    def sigmoid(self) -> None:
        pass
    
    def softmax(seld) -> None:
        pass