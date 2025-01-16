import os
import pandas as pd
from src.config import RAW_DATA_PATH, RAW_DATA_FILE, COLUMN_NAMES


def load_data() -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.

    Args:
        None
    Returns:
        pd.DataFrame: Data loaded into a DataFrame.
    """
    if not os.path.exists(f"{RAW_DATA_PATH}/{RAW_DATA_FILE}"):
        raise FileNotFoundError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' does not exist.")
    try:
        return pd.read_csv(f"{RAW_DATA_PATH}/{RAW_DATA_FILE}",
                           header=None,
                           names=COLUMN_NAMES)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' is empty.")
    except Exception as e:
        raise Exception(f"An error occured while reading the file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}': {e}")


def clean_data(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    DOES NOTHING FOR THE MOMENT BEING

    Args:
        None
    Returns:
        None
    """
    dataset_cleaned = raw_dataset
    # print(dataset_cleaned.head()) # TEST
    # print(dataset_cleaned.info()) # TEST
    # print(dataset_cleaned.describe()) # TEST
    # print(dataset_cleaned.dtypes) # TEST
    # print(dataset_cleaned.isnull().sum()) #TEST
    # print(dataset_cleaned['diagnosis'].value_counts()) # TEST
    return dataset_cleaned


def process_data() -> None:
    """
    Manage the data loading and preprocessing.

    Args:
        None    
    Return:
        None
    """
    raw_dataset = load_data()
    dataset_cleaned = clean_data(raw_dataset)
   