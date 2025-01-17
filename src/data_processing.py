import os
import pandas as pd
from dataclasses import dataclass, field
from src.config import RAW_DATA_PATH, RAW_DATA_FILE, COLUMN_NAMES, TRAIN_SIZE, RANDOM_SEED


@dataclass
class DataManager:
    full_dataset: pd.DataFrame = field(init=False)
    train_dataset: pd.DataFrame = field(init=False)
    val_dataset: pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        self.load_data()
        self.clean_data()
        self.divide_data()


    def load_data() -> None:
        """
        Load data from a CSV file into a DataFrame.

        Args:
            None
        Returns:
            None
        """
        file_path = os.path.join(RAW_DATA_PATH, RAW_DATA_FILE)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' does not exist.")
        try:
            full_dataset = pd.read_csv(f"{RAW_DATA_PATH}/{RAW_DATA_FILE}",
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
        self.train_dataset = self.full_dataset.sample(frac=TRAIN_SIZE, random_state=RANDOM_SEED)
        self.val_dataset = self.full_dataset.drop(self.train_dataset.index)
        