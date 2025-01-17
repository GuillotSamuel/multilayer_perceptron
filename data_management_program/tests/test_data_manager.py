# test_data_manager

import sys
import os
import pytest
import pandas as pd
from src.data_manager import DataManager
from config import RAW_DATA_PATH, RAW_DATA_FILE, COLUMN_NAMES, TRAIN_SIZE, RANDOM_SEED

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_manager import DataManager
from config import RAW_DATA_PATH, RAW_DATA_FILE, COLUMN_NAMES, TRAIN_SIZE, RANDOM_SEED


# Mocking the raw data path and file for testing
RAW_DATA_PATH = "tests/data"
RAW_DATA_FILE = "data.csv"
COLUMN_NAMES = ["col1", "col2", "col3"]
TRAIN_SIZE = 0.8
RANDOM_SEED = 42


@pytest.fixture
def mock_file_existence():
    """Fixture to mock the file existence check"""
    with patch("os.path.exists") as mocked_exists:
        mocked_exists.return_value = True
        yield mocked_exists


@pytest.fixture
def mock_read_csv():
    """Fixture to mock pandas.read_csv"""
    with patch("pandas.read_csv") as mocked_read_csv:
        yield mocked_read_csv


@pytest.fixture
def mock_data_manager(mock_file_existence, mock_read_csv):
    """Fixture to create an instance of DataManager with mocked methods"""
    # Mocking the return value of read_csv
    mock_read_csv.return_value = pd.DataFrame({
        "col1": [1, 2],
        "col2": [3, 4],
        "col3": [5, 6],
    })
    
    # Create the DataManager instance
    data_manager = DataManager()
    yield data_manager


def test_load_data(mock_data_manager, mock_read_csv):
    """Test for load_data method"""
    # Call the load_data method
    mock_data_manager.load_data()
    
    # Check if pandas.read_csv was called with the correct arguments
    mock_read_csv.assert_called_once_with(f"{RAW_DATA_PATH}/{RAW_DATA_FILE}",
                                          header=None,
                                          names=COLUMN_NAMES)


def test_load_data_file_not_found(mock_data_manager, mock_read_csv, mock_file_existence):
    """Test for load_data when file doesn't exist"""
    # Simulate the file not existing
    mock_file_existence.return_value = False
    
    # Expect a FileNotFoundError to be raised
    with pytest.raises(FileNotFoundError):
        mock_data_manager.load_data()


@patch("os.path.exists")
@patch("pandas.read_csv")
@patch.object(DataManager, "__post_init__", lambda x: None)  # Mock __post_init__ to prevent it from running
def test_load_data_empty_file(mock_read_csv, mock_exists):
    """
    Test the load_data method when the file is empty.
    """
    # Simulate file existence and reading of an empty file
    mock_exists.return_value = True
    mock_read_csv.side_effect = pd.errors.EmptyDataError  # Simulate EmptyDataError for an empty file

    data_manager = DataManager()  # Initialize DataManager without triggering __post_init__

    # Now test if the correct exception is raised
    with pytest.raises(ValueError):
        data_manager.load_data()


def test_clean_data(mock_data_manager):
    """Test for clean_data method"""
    # Just verify that clean_data doesn't raise errors or modify anything for now
    try:
        mock_data_manager.clean_data()
    except Exception as e:
        pytest.fail(f"clean_data raised an exception: {e}")


def test_divide_data(mock_data_manager):
    """Test for divide_data method"""
    # Check that the division of data occurs correctly
    mock_data_manager.full_dataset = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [6, 7, 8, 9, 10],
        "col3": [11, 12, 13, 14, 15],
    })
    
    mock_data_manager.divide_data()
    
    # Check if training and validation datasets are set
    assert mock_data_manager.train_dataset.shape[0] == int(5 * TRAIN_SIZE)
    assert mock_data_manager.val_dataset.shape[0] == 5 - int(5 * TRAIN_SIZE)
    

def test_data_manager(mock_data_manager):
    """Test the overall functionality of DataManager"""
    # Load data
    mock_data_manager.load_data()
    
    # Check the data loaded correctly
    assert mock_data_manager.full_dataset is not None
    assert len(mock_data_manager.full_dataset) == 2  # Based on mock data
    
    # Clean data (currently no functionality)
    mock_data_manager.clean_data()
    
    # Divide data
    mock_data_manager.divide_data()
    
    # Check if the data is divided properly
    assert mock_data_manager.train_dataset.shape[0] == int(2 * TRAIN_SIZE)
    assert mock_data_manager.val_dataset.shape[0] == 2 - int(2 * TRAIN_SIZE)
