import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LEARNING_RATE

class TrainingManager:
    
    def __init__(self) -> None:
        """
        Constructor.

        Args:
            None
        Returns:
            None
        """
        self.training_data = self.load_data(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        self.validation_data = self.load_data(f"{PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")
        

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
