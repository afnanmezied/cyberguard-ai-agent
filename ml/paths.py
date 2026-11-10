
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
