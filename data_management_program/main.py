from src.data_manager import DataManager
# from src.model import build_model
# from src.training import train_model
# from src.prediction import make_predictions

def main() -> None:
    """
    Main function to create DataManager class.

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
