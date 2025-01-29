import os
import sys
import pandas as pd
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, TRAIN_SIZE, RANDOM_SEED, COLUMN_NAMES


@dataclass
class DataManager:
    full_dataset: pd.DataFrame = field(init=False)
    training_dataset: pd.DataFrame = field(init=False)
    validation_dataset: pd.DataFrame = field(init=False)
    training_results: pd.DataFrame = field(init=False)
    validation_results: pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        self.load_data()
        self.clean_data()
        self.divide_data()
        self.save_splited_data()


    def load_data(self) -> None:
        """
        Load data from a CSV file into a DataFrame.
        Add column names.
        
        Args:
            None
        Returns:
            None
        """
        file_path = os.path.join(RAW_DATA_PATH, RAW_DATA_FILE)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' does not exist.")
        try:
            self.full_dataset = pd.read_csv(f"{RAW_DATA_PATH}/{RAW_DATA_FILE}",
                            header=None,
                            names=COLUMN_NAMES)
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' is empty.")
        except Exception as e:
            raise Exception(f"An error occured while reading the file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}': {e}")


    def clean_data(self) -> None:
        """
        DOES NOTHING FOR THE MOMENT BEING

        Args:
            None
        Returns:
            None
        """
        # print(dataset_cleaned.head()) # TEST
        # print(dataset_cleaned.info()) # TEST
        # print(dataset_cleaned.describe()) # TEST
        # print(dataset_cleaned.dtypes) # TEST
        # print(dataset_cleaned.isnull().sum()) #TEST
        # print(dataset_cleaned['diagnosis'].value_counts()) # TEST
        pass # TEST


    def divide_data(self) -> None:
        """
        Divide the cleaned dataset into two new datasets:
            - training_dataset
            - validation_dataset
        
        Args:
            None
        Returns:
            None
        """
        self.training_dataset = self.full_dataset.sample(frac=TRAIN_SIZE, random_state=RANDOM_SEED)
        self.validation_dataset = self.full_dataset.drop(self.training_dataset.index)
         
        self.training_results = self.training_dataset['diagnosis'].copy()
        self.validation_results = self.validation_dataset['diagnosis'].copy()
        
        self.training_dataset = self.training_dataset.drop(columns=['id', 'diagnosis'])
        self.validation_dataset = self.validation_dataset.drop(columns=['id', 'diagnosis'])


    def save_splited_data(self) -> None:
        """
        Save the two new datasets in two csv files.
        Location -> data/processed
        
        Args:
            None
        Returns:
            None
        """
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        self.training_dataset.to_csv(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}", index=False)
        self.validation_dataset.to_csv(f"{PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}", index=False)
        self.training_results.to_csv(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}", index=False)
        self.validation_results.to_csv(f"{PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}", index=False)

        print(f"Training dataset saved to {PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        print(f"Validation dataset saved to {PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")
        print(f"Validation dataset saved to {PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")
        print(f"Validation dataset saved to {PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")