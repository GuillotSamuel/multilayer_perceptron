# config.py for the Wisconsin Diagnostic Breast Cancer (WDBC) case

# Data files configuration
RAW_DATA_PATH = "data/raw"
RAW_DATA_FILE = "numbers.csv"
RAW_DATA_IMAGES_PATH = "numbers"
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
EARLY_STOPPING_LIMIT = 0
NORM_METHOD = 'pixelNorm'

# Optimizer Parameters
ADAM = {
    'learning_rate': LEARNING_RATE,
    'beta_1': 0.9,  # Momentum term, usually around 0.9
    'beta_2': 0.999,  # Squared gradient term, usually 0.999
    'epsilon': 1e-07,  # Small constant to avoid division by zero
}

# Regularization Parameters
DROPOUT = 0.2