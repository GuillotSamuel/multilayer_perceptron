from src.data_processing import load_data
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
    data = load_data("data/raw/data.csv")
    

if __name__ == "__main__":
    main()
