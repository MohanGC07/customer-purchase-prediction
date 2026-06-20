"""
Model loading and prediction logic for bank deposit subscription.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Tuple

from .utils import MODEL_PATH, SCALER_PATH, COLUMNS_PATH
from .preprocess import engineer_features, encode_features, align_columns


class DepositPredictor:
    """
    Production-ready predictor for bank term deposit subscription.
    
    Loads trained model artifacts and provides a clean prediction interface
    for both single and batch predictions.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.train_columns = None
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model, scaler, and column reference from disk."""
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        loaded_columns = joblib.load(COLUMNS_PATH)
        self.train_columns = list(loaded_columns) if loaded_columns is not None else []
    
    def preprocess(self, input_dict: dict) -> np.ndarray:
        """
        Full preprocessing pipeline: feature engineering → encoding → column alignment → scaling.
        
        Args:
            input_dict: Raw customer features
            
        Returns:
            Scaled numpy array ready for prediction
        """
        # Feature engineering
        df = engineer_features(input_dict)
        
        # One-hot encoding
        categorical_cols = [
            "job", "marital", "education", "housing", "loan",
            "contact", "poutcome", "age_group", "balance_category", "contact_intensity"
        ]
        df_encoded = encode_features(df, categorical_cols)
        
        # Align columns with training data
        df_aligned = align_columns(df_encoded, self.train_columns)
        
        # Scale features
        return self.scaler.transform(df_aligned)
    
    def predict(self, input_dict: dict) -> Tuple[int, float]:
        """
        Make prediction on raw customer data.
        
        Args:
            input_dict: Raw customer features
            
        Returns:
            Tuple of (prediction: 0 or 1, probability: float)
        """
        processed = self.preprocess(input_dict)
        prediction = self.model.predict(processed)[0]
        probability = self.model.predict_proba(processed)[0][1]
        return int(prediction), float(probability)