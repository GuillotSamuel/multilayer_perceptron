import os
import pandas as pd


class Utils:
    
    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        return pd.read_csv(file_path)


    @staticmethod
    def save_model() -> None:
        pass
