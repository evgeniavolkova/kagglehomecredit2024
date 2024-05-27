"""Configuration file."""

import os
from pathlib import Path

# Determine the execution environment based on environment variables
KAGGLE = 'KAGGLE_URL_BASE' in os.environ
COLAB = 'COLAB_GPU' in os.environ

# Define base paths for different environments
base_paths = {
    "COLAB": Path("/content/drive/MyDrive/Kaggle/HomeCredit"),
    "KAGGLE": Path("/kaggle/input"),
}

# Set paths based on the environment
if COLAB:
    base_path = base_paths["COLAB"]
    PATH_DATA = base_path / "data"
    PATH_DATA_PROC = base_path / "data/processed"
    PATH_MODELS = base_path / "models"
    PATH_FEATURES = base_path / "features"
    PATH_CODE = base_path / "dist/homecredit-0.1-py3-none-any.whl"
elif KAGGLE:
    base_path = base_paths["KAGGLE"] 
    PATH_DATA = base_path / "home-credit-credit-risk-model-stability"
    PATH_DATA_PROC = base_path / "homecredit-data-processed"
    PATH_MODELS = base_path / "homecredit-models"
    PATH_FEATURES = base_path / "homecredit-features"
    PATH_CODE = base_path / "homecredit-code/homecredit-0.1-py3-none-any.whl"
else:
    raise ValueError("Unknown environment")

PATHS_DATA = {
    "train": PATH_DATA / "csv_files/train",
    "test": PATH_DATA / "csv_files/test",
}

# Kaggle
KAGGLE_USERNAME = "eivolkova"

# Random seed
RANDOM_SEED = 42

# Data column names
COL_TARGET = "target"
COL_ID = "case_id"
COL_WEEK = "WEEK_NUM"
COL_DATE = "date_decision"