
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Original dataset
PHIUSIIL_RAW = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"

# Cleaned dataset
PHIUSIIL_CLEAN = PROCESSED_DIR / "phiusiil_clean.csv"

# Extracted features dataset
PHIUSIIL_FEATURES = PROCESSED_DIR / "phiusiil_features.csv"

# Split datasets
TRAIN_DATA = PROCESSED_DIR / "train.csv"
VALIDATION_DATA = PROCESSED_DIR / "validation.csv"
TEST_DATA = PROCESSED_DIR / "test.csv"

# Trained Random Forest model
RANDOM_FOREST_MODEL = BASE_DIR / "models" / "random_forest" / "random_forest.joblib"