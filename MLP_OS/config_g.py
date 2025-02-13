# config.py

# Data files configuration
RAW_DATA_PATH = "data/raw"
RAW_DATA_FILE = "data.csv"
PROCESSED_DATA_PATH = "data/processed"
TRAINING_DATA_FILE = "training.csv"
VALIDATION_DATA_FILE = "validation.csv"
TRAINING_RESULTS_FILE = "training_results.csv"
VALIDATION_RESULTS_FILE = "validation_results.csv"
LOGS_FOLDER = "logs"
LOSS_LOGS_FILE = "training_logs.png"
MODEL_PATH = "model"
MODEL_FILE = "model"
PREDICTION_PATH = "prediction"
PREDICTION_FILE = "prediction.csv"

# Data division parameters
TRAIN_SIZE = 0.80
RANDOM_SEED = 42

# Training parameters
# NETWORK
LAYER = [24, 24, 12]
EPOCHS = 84
LOSS = 'binaryCrossentropy'
LEARNING_RATE = 0.1
BATCH_SIZE = 50
