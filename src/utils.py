"""
Utility functions and configuration for the Bank Deposit Prediction project.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_model_path(filename: str) -> str:
    """Return full path to a model file."""
    return str(get_project_root() / "models" / filename)


def get_data_path(filename: str, folder: str = "raw") -> str:
    """Return full path to a data file."""
    return str(get_project_root() / "data" / folder / filename)


# Model artifacts
MODEL_PATH = get_model_path("random_forest_model.pkl")
SCALER_PATH = get_model_path("scaler.pkl")
COLUMNS_PATH = get_model_path("X_train_columns.pkl")

# Data paths
RAW_DATA_PATH = get_data_path("bank.csv", folder="raw")
PROCESSED_DATA_PATH = get_data_path("bank_cleaned.csv", folder="processed")