import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, TRAIN_SIZE, RANDOM_SEED, RAW_DATA_IMAGES_PATH


@dataclass
class DataManager:
    full_dataset: pd.DataFrame = field(init=False)
    training_dataset: pd.DataFrame = field(init=False)
    validation_dataset: pd.DataFrame = field(init=False)
    training_results: pd.DataFrame = field(init=False)
    validation_results: pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        self.load_data()
        print("Data loaded.")
        self.clean_data()
        print("Data cleaned.")
        self.divide_data()
        print("Data divided.")
        self.save_splited_data()


    def load_data(self) -> None:
        """ 
        Load CSV and PNG files into a DataFrame

        Args:
            None
        Returns:
            None
        """
        data = []
        try:
            file_path = os.path.join(RAW_DATA_PATH, RAW_DATA_FILE)     
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file '{RAW_DATA_FILE}' at the path '{RAW_DATA_PATH}' does not exist.")

            image_metadata = pd.read_csv(file_path)

            for _, row in image_metadata.iterrows():
                origin = row['origin']
                group = row['group']
                label = row['label']
                image_path = row['file']
                
                image_full_path = f"{RAW_DATA_PATH}/numbers/{image_path}"
                
                if not os.path.exists(image_full_path):
                    print(f"Warning: Image file '{image_full_path}' not found, skipping.")
                    continue
                
                with Image.open(image_full_path) as img:
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                    img = img.convert('L')
                    pixels = np.array(img).flatten()
                
                pixel_columns = {f'pixel_{i}': pixel for i, pixel in enumerate(pixels)}
                data.append({
                    **pixel_columns,
                    'origin': origin,
                    'group': group,
                    'label': label,
                    'file': image_full_path,
                })
            
            self.full_dataset = pd.DataFrame(data)
            print("All datas have been loaded.")
        
        except Exception as e:
            raise Exception(f"An error occurred while loading PNG files: {e}")


    def clean_data(self) -> None:
        """
        DOES NOTHING FOR THE MOMENT BEING

        Args:
            None
        Returns:
            None
        """
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

        self.training_results = self.training_dataset['label'].copy()
        self.validation_results = self.validation_dataset['label'].copy()
        
        self.training_dataset = self.training_dataset.drop(columns=['origin', 'group', 'label', 'file'])
        self.validation_dataset = self.validation_dataset.drop(columns=['origin', 'group', 'label', 'file'])


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
        print(f"Training dataset saved to {PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")

        self.validation_dataset.to_csv(f"{PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}", index=False)
        print(f"Validation dataset saved to {PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")

        self.training_results.to_csv(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}", index=False)
        print(f"Training results saved to {PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")

        self.validation_results.to_csv(f"{PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}", index=False)
        print(f"Validation results saved to {PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}")


if __name__ == "__main__":
    try:
        DataManager()

    except Exception as e:
        print(f"Error: {e}")
