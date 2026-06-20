"""
Model Performance Metrics Page
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_PATH

st.set_page_config(page_title="Model Metrics", page_icon="📊")

st.title("📊 Model Performance Metrics")

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

try:
    model = load_model()
    df = load_data()
    
    # Prepare features (adjust based on your actual processed data)
    # This is a simplified version - you may need to adapt to your actual columns
    target_col = "deposit" if "deposit" in df.columns else "y"
    
    if target_col not in df.columns:
        st.warning("Target column not found in processed data. Please ensure your processed dataset includes the target variable.")
        st.stop()
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical encoding if needed
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Predictions
    y_pred = model.predict(X_encoded)
    y_prob = model.predict_proba(X_encoded)[:, 1]
    
    # Metrics
    report = classification_report(y, y_pred, output_dict=True)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.3f}")
    col2.metric("Precision", f"{report['1']['precision']:.3f}")
    col3.metric("Recall", f"{report['1']['recall']:.3f}")
    col4.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
    
    st.divider()
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Deposit", "Deposit"],
                yticklabels=["No Deposit", "Deposit"])
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)
    
    st.divider()
    
    # Classification Report
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))
    
    st.divider()
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
    
    st.metric("AUC Score", f"{roc_auc:.3f}")

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.info("Please ensure your model and processed data files are available.")