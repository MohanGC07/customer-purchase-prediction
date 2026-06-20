"""
Data preprocessing and feature engineering for bank deposit prediction.
"""

import pandas as pd
from typing import Dict, Any


def create_age_group(age: int) -> str:
    """Categorize age into life-stage groups."""
    if age <= 30:
        return "Young Adult"
    elif age <= 45:
        return "Adult"
    elif age <= 60:
        return "Middle-Aged"
    else:
        return "Senior"


def create_balance_category(balance: float) -> str:
    """Categorize account balance into tiers."""
    if balance < 1000:
        return "Low"
    elif balance <= 5000:
        return "Medium"
    else:
        return "High"


def create_contact_intensity(campaign: int) -> str:
    """Categorize marketing contact intensity."""
    if campaign <= 2:
        return "Low"
    elif campaign <= 5:
        return "Medium"
    else:
        return "High"


def engineer_features(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply feature engineering to raw customer input.
    
    Args:
        input_dict: Dictionary with raw customer features
        
    Returns:
        DataFrame with engineered features (age_group, balance_category, contact_intensity)
    """
    df = pd.DataFrame([input_dict])
    
    df["age_group"] = df["age"].apply(create_age_group)
    df["balance_category"] = df["balance"].apply(create_balance_category)
    df["contact_intensity"] = df["campaign"].apply(create_contact_intensity)
    
    return df


def encode_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    One-hot encode categorical features.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with one-hot encoded features
    """
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def align_columns(df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """
    Align DataFrame columns with training data to prevent feature mismatch.
    
    Args:
        df: Input DataFrame
        train_columns: List of columns from training data
        
    Returns:
        DataFrame aligned with training columns
    """
    return df.reindex(columns=train_columns, fill_value=0)