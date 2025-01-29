import os
import sys
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE

class TrainingManager:
    def __init__(self) -> None:
        self.num_outputs = self.count_outputs()
        
        args = self.parse_arguments()

        self.layers = args.layer
        self.activation = args.activation
        self.weight_initializer = args.weight_initializer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.train()
        self.validate()


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

        return args


    def load_file(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        return pd.read_csv(file_path)
    
    
    def count_outputs(self) -> int:
        training_results = self.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")
        validation_results = self.load_file(f"{PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}")
        
        all_results = pd.concat([training_results, validation_results])

        num_outputs = all_results['diagnosis'].nunique()
        
        return num_outputs


    def standardize_data(self, dataset: pd.DataFrame, results: pd.DataFrame) -> tuple:
        mean, std = dataset.mean(), dataset.std()
        X = (dataset - mean) / std.replace(0, 1)

        categories = list(results['diagnosis'].unique())
        category_map = {category: i for i, category in enumerate(categories)}
        results_indices = results['diagnosis'].map(category_map).to_numpy()

        Y = np.zeros((len(results_indices), self.num_outputs), dtype=int)
        Y[np.arange(len(results_indices)), results_indices] = 1

        return X.to_numpy(), Y


    def heUniform(self, input_size: int, layer_size: int) -> np.ndarray: # TO CHECK
        limit = np.sqrt(6 / input_size)
        return np.random.uniform(-limit, limit, (input_size, layer_size))


    def gaussian_randomNormal(self, input_size: int, layer_size: int) -> np.ndarray:
        return np.random.randn(input_size, layer_size) * 0.01


    def initialize_model(self, X: tuple) -> dict:
        model = {}
        input_size = X.shape[1]

        for i, layer_size in enumerate(self.layers):
            if self.weight_initializer[i] == "heUniform":
                model[f"W{i+1}"] = self.heUniform(input_size, layer_size)
            elif self.weight_initializer[i] == "randomNormal":
                model[f"W{i+1}"] = self.gaussian_randomNormal(input_size, layer_size)
            model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size

        return model


    def relu(self, Z: np.ndarray) -> np.ndarray: # TO CHECK
        return np.maximum(0, Z)


    def sigmoid(self, Z: np.ndarray) -> np.ndarray: # TO CHECK
        return 1 / (1 + np.exp(-Z))


    def softmax(self, Z: np.ndarray) -> np.ndarray: # TO CHECK
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


    def forward_propagation(self, model: dict, X: np.ndarray) -> np.ndarray:
        A = X

        for i in range(len(self.layers)):
            Z = np.dot(A, model[f"W{i+1}"]) + model[f"b{i+1}"]
            if self.activation[i] == "sigmoid":
                A = self.sigmoid(Z)
            elif self.activation[i] == "softmax":
                A = self.softmax(Z)

        return A
    
    
    def binary_crossentropy(self, Y_true: np.ndarray, Y_predicted: np.ndarray) -> float:
        m = Y_true.shape[0]

        epsilon = 1e-10
        Y_pred = np.clip(Y_predicted, epsilon, 1 - epsilon)

        loss = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))

        return loss


    def backpropagate(self, X: np.ndarray, Y: np.ndarray, model: dict) -> dict:
        
        return
        
        
    def update_weights_and_display(self, model: dict, gradients: dict, epoch: int, X: np.ndarray, Y: np.ndarray) -> None:
        
        return


    def train(self) -> None:
        dataset = self.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        results = self.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")
        X, Y = self.standardize_data(dataset, results)
        model = self.initialize_model(X)
        for epoch in range(self.epochs):
            A = self.forward_propagation(model, X)
            loss = self.binary_crossentropy(Y, A)

    def validate(self) -> None:
        pass
