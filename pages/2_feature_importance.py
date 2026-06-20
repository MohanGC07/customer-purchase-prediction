"""
Feature Importance Analysis Page
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import MODEL_PATH, COLUMNS_PATH

st.set_page_config(page_title="Feature Importance", page_icon="📈")

st.title("📈 Feature Importance")

try:
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_names = list(columns) if hasattr(columns, '__iter__') else columns
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    # Top 15 features
    top_n = st.slider("Number of top features to display", 5, min(30, len(importance_df)), 15)
    top_features = importance_df.head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, y="Feature", x="Importance", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    
    st.divider()
    
    # Data table
    st.subheader("Feature Importance Scores")
    st.dataframe(
        importance_df.style.format({"Importance": "{:.4f}"})
        .bar(subset=["Importance"], color="#29b5e8")
    )
    
    st.info("""
    **Interpretation:** Features with higher importance scores have greater influence 
    on the model's prediction. This helps identify which customer attributes 
    (age, balance, contact history) most strongly indicate deposit subscription likelihood.
    """)

except Exception as e:
    st.error(f"Error loading model: {e}")