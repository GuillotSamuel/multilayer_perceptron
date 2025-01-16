# config.py

RAW_DATA_PATH = "data/raw"
RAW_DATA_FILE = "data.csv"
PROCESSED_DATA_PATH = "data/processed"
TRAINING_DATA_FILE = "training.csv"
VALIDATION_DATA_FILE = "validation.csv"



COLUMN_NAMES = [
    "id", "diagnosis",
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se", "worst_radius", "worst_texture", "worst_perimeter",
    "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]
