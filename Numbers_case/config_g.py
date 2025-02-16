# config.py for the Number recognition case

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
TRAIN_SIZE = 0.80 # Default training size
RANDOM_SEED = 42 # Random seed for random train-test split

# Training parameters
# NETWORK
LAYER = [24, 24, 12] # Default number of neurones in each hidden layer
EPOCHS = 84 # Default number of epochs
LOSS = 'binaryCrossentropy'# Default loss function
LEARNING_RATE = 0.1 # Default learning rate
BATCH_SIZE = 50 # Default batch size
NORM_METHOD = 'pixelNorm' # Default normalization method

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 3 # Default early stopping limit
EARLY_STOPPING_MIN_DELTA = 1e-6 # Default early stopping min delta

# Optimizer Parameters
ADAM = {
    'learning_rate': LEARNING_RATE,
    'beta_1': 0.9,  # Momentum term, usually around 0.9
    'beta_2': 0.999,  # Squared gradient term, usually 0.999
    'epsilon': 1e-07,  # Small constant to avoid division by zero
}

# Regularization Parameters
DROPOUT_RATE = 0.2 # Fraction of the neurones units to drop randomly on each hidden layer