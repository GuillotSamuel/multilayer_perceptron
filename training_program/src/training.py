import os
import sys
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE

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

        args.layer.append(1)

        if len(args.layer) > 1:
            args.activation = ["sigmoid"] * (len(args.layer) - 1) + ["softmax"]
            args.weight_initializer = ["heUniform"] * (len(args.layer) - 1) + ["heUniform"]
        else:
            args.activation = ["sigmoid"]
            args.weight_initializer = ["heUniform"]

        return args


    def load_data(self, data_path: str, results_path: str) -> pd.DataFrame:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file '{data_path}' does not exist.")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"The file '{results_path}' does not exist.")

        return pd.read_csv(data_path), pd.read_csv(results_path)


    def standardize_data(self, dataset: pd.DataFrame, results: pd.DataFrame) -> tuple:
        mean, std = dataset.mean(), dataset.std()
        X = (dataset - mean) / std

        categories = list(results['diagnosis'].unique())
        category_map = {category: i for i, category in enumerate(categories)}
        results_indices = results['diagnosis'].map(category_map).to_numpy()

        Y = np.zeros((len(results_indices), len(categories)), dtype=int)
        Y[np.arange(len(results_indices)), results_indices] = 1

        return X.to_numpy(), Y


    def initialize_model(self, X: tuple) -> dict:
        model = {}
        input_size = X.shape[1]

        for i, layer_size in enumerate(self.layers):
            model[f"W{i+1}"] = np.random.randn(input_size, layer_size) * 0.01
            model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size

        print(f"X : {X.shape}\n\n")

        for key, value in model.items():
            print(f"{key}: {value.shape}\n")

        return model


    def sigmoid(self, Z: np.ndarray):
        return 1 / (1 + np.exp(-Z))


    def forward_propagation(self, model: dict, X: np.ndarray) -> np.ndarray:
        A = X

        for i in range(len(self.layers)):
            Z = np.dot(A, model[f"W{i+1}"]) + model[f"b{i+1}"]
            A = self.sigmoid(Z)

        return A
    
    
    def binary_crossentropy(self):
        pass

    def train(self) -> None:
        dataset, results = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}",
                                          f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")
        X, Y = self.standardize_data(dataset, results)
        model = self.initialize_model(X)
        for epoch in range(self.epochs):
            A = self.forward_propagation(model, X)


    def validate(self) -> None:
        pass
