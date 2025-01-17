from src.data_manager import DataManager


def main() -> None:
    """
    Main function to create DataManager class.

    Args:
        None
    Return:
        None
    """
    try:
        DataManager()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
