from src.data_processing import DataManager
# from src.model import build_model
# from src.training import train_model
# from src.prediction import make_predictions

def main() -> None:
    """
    Main function to:
        1. Load and preprocess data
        2. Build the model
        3. Train and save the model
        4. Make predictions and display results

    Args:
        None
    Return:
        None
    """
    try:
        data = DataManager()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
