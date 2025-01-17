from src.training import TrainingManager


def main() -> None:
    """
    Main function to create DataManager class.

    Args:
        None
    Return:
        None
    """
    try:
        TrainingManager()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
