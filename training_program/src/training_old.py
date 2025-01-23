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
        """
        Constructor.

        Args:
            None
        Returns:
            None
        """
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
        self.X, self.results = self.standardize_data()

        self.initialize_model()
        self.log_data_training_mse = []
        self.log_data_training_mae = []
        self.train()


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
        training_features = self.training_dataset.drop(columns=['id', 'diagnosis'])
        training_labels_diag = self.training_dataset['diagnosis']

        mean = training_features.mean()
        std = training_features.std()
        standardized_features = (training_features - mean) / std

        x_training_dataset = standardized_features
        y_training_dataset = training_labels_diag

        return x_training_dataset, y_training_dataset


    def initialize_model(self) -> None:
        """
        Initialize the neural network model by setting up weights and biases for each layer.

        Args:
            None
        Returns:
            None
        """
        self.model = {}
        input_size = self.X.shape[1]
        
        for i, layer_size in enumerate(self.layers):
            self.model[f"W{i+1}"] = np.random.randn(input_size, layer_size) * 0.01
            self.model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size


    def train(self) -> None:
        """
        
        """
        for epoch in range(self.epochs):
            predictions = self.forward_propagation()
            mse = self.mean_squarred_error(predictions)
            # mae = self.mean_absolute_error(predictions)
            # gradients = self.backward_propagation(self.X, self.Y, predictions)
            # self.update_parameters(gradients)
        #     self.log_data_training_mse.append([epoch + 1, mse])
        #     self.log_data_training_mae.append([epoch + 1, mae])
        # self.save_loss_logs()


    def forward_propagation(self) -> np.ndarray:
        """
        Perform forward propagation through the neural network.

        Args:
            None
        Returns:
            np.ndarray: The final output of the network after forward propagation.
        """
        A = self.X
        for i in range(1, len(self.layers) + 1):
            W = self.model[f"W{i}"]
            b = self.model[f"b{i}"]
            Z = np.dot(A, W) + b
            A = self.sigmoid(Z)
        return A


    def sigmoid(self, Z) -> np.ndarray:
        """
        Compute the sigmoid activation function for each element in Z.

        The sigmoid function is a type of activation function that maps input values 
        to a range between 0 and 1. It's defined as 1 / (1 + exp(-Z)).

        Args:
        Z (np.ndarray): The input array, which can be a matrix or a vector.

        Returns:
        np.ndarray: The result of applying the sigmoid function element-wise to Z.
        """
        return 1 / (1 + np.exp(-Z))


    def softmax(self, Z) -> np.ndarray:
        """
        Compute the softmax activation for each row in Z.

        The softmax function is often used in multi-class classification problems to 
        convert the input into probabilities that sum to 1 for each row.

        Args:
        Z (np.ndarray): The input array, typically with shape (n_samples, n_classes).

        Returns:
        np.ndarray: The output probabilities of the softmax function, same shape as Z.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


    def reLU(self, Z) -> np.ndarray:
        """
        Compute the ReLU (Rectified Linear Unit) activation function for each element in Z.

        The ReLU function outputs the input directly if it is positive, 
        otherwise, it will output zero. It's commonly used in hidden layers of neural networks.

        Args:
        Z (np.ndarray): The input array, can be a matrix or vector.

        Returns:
        np.ndarray: The result of applying the ReLU function element-wise to Z.
        """
        return np.maximum(0, Z)


    def mean_squarred_error(self, predictions):
        """
        Compute the loss using mean squared error (MSE).
        """
        length = self.Y.shape[0]
        print(f"{predictions}\n\n{self.Y}")
        loss = np.sum((predictions - self.Y) ** 2) / (2 * length)
        return loss


    def mean_absolute_error(self, predictions):
        """
        Compute the loss using mean absolute error (MAE).
        """
        mae = 0
        length = self.Y.shape[0]

        for i in range(length):
            mae += abs(predictions[i] - self.Y[i])
        mae /= length
        return mae


    def backward_propagation(self, X, Y, predictions):
        """
        
        """


    def update_parameters(self, gradients):
        """
        
        """


    def save_loss_logs(self) -> None:
        """
        Saving logs into a CSV file.
        """
        if self.log_data_training_mse and self.log_data_training_mae:
            log_data = []

            for epoch in range(len(self.log_data_training_mse)):
                mse = self.log_data_training_mse[epoch][1]
                mae = self.log_data_training_mae[epoch][1]
                log_data.append([self.log_data_training_mse[epoch][0], mse, mae])

            log_df = pd.DataFrame(log_data, columns=["Epoch", "MSE", "MAE"])

            if not os.path.exists(LOGS_FOLDER):
                os.makedirs(LOGS_FOLDER)

            log_df.to_csv(f"{LOGS_FOLDER}/{LOSS_LOGS_FILE}", index=False)
            print(f"Logs saved in {LOGS_FOLDER}/{LOSS_LOGS_FILE}")
        else:
            print(f"No datas to put in {LOGS_FOLDER}/{LOSS_LOGS_FILE}.")
