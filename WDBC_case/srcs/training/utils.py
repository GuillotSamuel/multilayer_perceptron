import os
import pandas as pd
import numpy as np


class Utils:
    
    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        return pd.read_csv(file_path)


    @staticmethod
    def save_model(model_path: str, model_file: str, parameters: dict, config: dict) -> None:
        """Save the model and its configuration."""
        os.makedirs(model_path, exist_ok=True)
        
        save_data = {
            'parameters': {k: v.tolist() for k, v in parameters.items()},
            'config': config
        }
        
        full_path = os.path.join(model_path, model_file)
        np.save(full_path, save_data, allow_pickle=True)
        print(f"\nModel saved successfully to {full_path}")
        
        
    @staticmethod
    def load_model(model_path: str, model_file: str) -> tuple:
        """ Load a model and its configuration. """
        full_path = os.path.join(model_path, model_file)
        
        if not os.path.exists(full_path + '.npy'):
            raise FileNotFoundError(f"No model file found at {full_path}")
        
        save_data = np.load(full_path + '.npy', allow_pickle=True).item()
        
        parameters = {k: np.array(v) for k, v in save_data['parameters'].items()}
        config = save_data['config']
        
        print(f"\nModel loaded successfully from {full_path}")
        return parameters, config