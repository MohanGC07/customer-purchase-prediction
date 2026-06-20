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

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import MODEL_PATH, SCALER_PATH, PROCESSED_DATA_PATH

st.set_page_config(page_title="Model Metrics", page_icon="📊")

st.title("📊 Model Performance Metrics")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)

try:
    model = load_model()
    df = load_data()
    
    # Auto-detect target column
    possible_targets = ["deposit", "y", "target", "subscribed", "response", "label"]
    target_col = next((col for col in possible_targets if col in df.columns), None)
    
    if target_col is None:
        # Try to find binary column
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                break
    
    if target_col is None:
        st.error("❌ Could not detect target column in processed data.")
        st.stop()
    
    st.success(f"✅ Target column detected: **{target_col}**")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert target to numeric if needed
    if y.dtype == object:
        y = y.map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).fillna(y)
        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X.copy()
    
    # Align with model features
    if hasattr(model, "feature_names_in_"):
        model_cols = list(model.feature_names_in_)
        X_encoded = X_encoded.reindex(columns=model_cols, fill_value=0)
    
    # Scale
    try:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_encoded)
    except Exception:
        X_scaled = X_encoded.values
    
    # Predictions
    y_pred = model.predict(X_scaled)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)
        y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
    else:
        y_prob = y_pred.astype(float)
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    pos_label = "1" if "1" in report else str(max(set(report.keys()) - {"accuracy", "macro avg", "weighted avg"}))
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.3f}")
    col2.metric("Precision", f"{report[pos_label]['precision']:.3f}")
    col3.metric("Recall", f"{report[pos_label]['recall']:.3f}")
    col4.metric("F1-Score", f"{report[pos_label]['f1-score']:.3f}")
    
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
    
    # Classification Report Table
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))
    
    st.divider()
    
    # ROC Curve
    try:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
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
    except Exception as roc_err:
        st.warning(f"Could not generate ROC curve: {roc_err}")

except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())