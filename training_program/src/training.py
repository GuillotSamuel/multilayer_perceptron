import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LEARNING_RATE, MINIBATCH_SIZE, LAYERS_SIZE

class TrainingManager:
    
    def __init__(self) -> None:
        """
        Constructor.

        Args:
            None
        Returns:
            None
        """
        self.training_dataset = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        self.num_layers = len(LAYERS_SIZE)
        self.sizes_layers = LAYERS_SIZE
        self.standardize_data()


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
        training_labels_id = self.training_dataset['id']
        training_labels_diag = self.training_dataset['diagnosis']

        mean = training_features.mean()
        std = training_features.std()
        standardized_features = (training_features - mean) / std
        self.training_dataset = pd.concat([training_labels_id, training_labels_diag, standardized_features], axis=1)


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
