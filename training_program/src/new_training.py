
import os
import sys
import pandas as pd
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE


class TrainingManager:
    def __init__(self) -> None:
        args = self.parse_arguments()
        self.layers = args.layer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.training_dataset = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        self.X, self.Y = self.standardize_data()

        self.initialize_model()
        self.train()
        self.save_model()
        self.validate()


    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Training Manager Configuration")
        parser.add_argument("--layer", type=int, nargs="+", default=LAYER, help="Size of each layer.")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs.")
        parser.add_argument("--loss", type=str, choices=["binaryCrossentropy"], default=LOSS, help="Loss function.")
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate.")
        return parser.parse_args()

    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        return pd.read_csv(file_path)

    def standardize_data(self) -> tuple:
        features = self.training_dataset.drop(columns=['id', 'diagnosis'])
        labels = self.training_dataset['diagnosis']

        mean, std = features.mean(), features.std()
        X = (features - mean) / std
        Y = labels.map({'M': 1, 'B': 0})  # Binary encoding: M = 1 (cancer), B = 0 (no cancer)

        return X.to_numpy(), Y.to_numpy().reshape(-1, 1)

    def initialize_model(self) -> None:
        self.model = {}
        input_size = self.X.shape[1]

        for i, layer_size in enumerate(self.layers):
            self.model[f"W{i+1}"] = np.random.randn(input_size, layer_size) * 0.01
            self.model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def binary_crossentropy_loss(self, predictions: np.ndarray) -> float:
        epsilon = 1e-15  # To avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(self.Y * np.log(predictions) + (1 - self.Y) * np.log(1 - predictions))

    def forward_propagation(self) -> tuple:
        activations = [self.X]
        Z_values = []

        for i in range(1, len(self.layers) + 1):
            Z = np.dot(activations[-1], self.model[f"W{i}"]) + self.model[f"b{i}"]
            Z_values.append(Z)
            activation = self.sigmoid(Z)
            activations.append(activation)

        return activations, Z_values

    def backward_propagation(self, activations: list, Z_values: list) -> None:
        m = self.X.shape[0]
        dA = -(self.Y / activations[-1]) + ((1 - self.Y) / (1 - activations[-1]))
        grads = {}

        for i in reversed(range(1, len(self.layers) + 1)):
            dZ = dA * self.sigmoid_derivative(Z_values[i-1])
            grads[f"dW{i}"] = np.dot(activations[i-1].T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.model[f"W{i}"].T)

        for i in range(1, len(self.layers) + 1):
            self.model[f"W{i}"] -= self.learning_rate * grads[f"dW{i}"]
            self.model[f"b{i}"] -= self.learning_rate * grads[f"db{i}"]


    def train(self) -> None:
        for epoch in range(self.epochs):
            activations, Z_values = self.forward_propagation()
            loss = self.binary_crossentropy_loss(activations[-1])
            self.backward_propagation(activations, Z_values)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")


    def save_model(self) -> None:
        os.makedirs(MODEL_PATH, exist_ok=True)
        model_data = {key: self.model[key] for key in self.model}
        np.savez(f"{MODEL_PATH}/{MODEL_FILE}", **model_data)
        print(f"Model saved at {MODEL_PATH}/{MODEL_FILE}.")
        
        
    def validate(self) -> None:
        validation_data = self.load_data(f"{PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")
        X_val, Y_val = self.standardize_data(validation_data)
        
        