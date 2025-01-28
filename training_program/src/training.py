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
        
        self.train()
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
    
    
    def standardize_data(self, dataset: pd.DataFrame) -> tuple:
        features = dataset.drop(columns=['id', 'diagnosis'])
        labels = dataset['diagnosis']

        mean, std = features.mean(), features.std()
        X = (features - mean) / std
        Y = labels.map({'M': 1, 'B': 0})

        return X.to_numpy(), Y.to_numpy().reshape(-1, 1)
    
    
    def initialize_model(self, X: tuple) -> dict:
        model = {}
        input_size = X.shape[1]
        
        for i, layer_size in enumerate(self.layers):
            model[f"W{i+1}"] = np.random.randn(input_size, layer_size) * 0.01
            model[f"b{i+1}"] = np.zeros((1, layer_size))
            input_size = layer_size
            
        model[f"W{len(self.layers)+1}"] = np.random.randn(input_size, 1) * 0.01
        model[f"b{len(self.layers)+1}"] = np.zeros((1, 1))
        
        print(f"X : {X.shape}\n\n")
        
        for key, value in model.items():
            print(f"{key}: {value.shape}\n")

        return model


    def train(self) -> None:
        dataset = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        X, Y = self.standardize_data(dataset)
        model = self.initialize_model(X)
        for epoch in range(self.epochs):
            pass
        

    def validate(self) -> None:
        pass