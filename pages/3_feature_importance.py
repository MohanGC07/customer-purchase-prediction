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
    
    # Convert columns to list
    if hasattr(columns, 'tolist'):
        feature_names = columns.tolist()
    elif hasattr(columns, '__iter__'):
        feature_names = list(columns)
    else:
        feature_names = [str(columns)]
    
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        st.error("This model does not support feature importance.")
        st.stop()
    
    # Handle length mismatch
    if len(importances) != len(feature_names):
        st.warning(f"Feature count mismatch: model={len(importances)}, columns file={len(feature_names)}")
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            st.info(f"Using model's feature names: {len(feature_names)} features")
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    # Slider
    top_n = st.slider("Number of top features", min_value=5, 
                      max_value=min(30, len(importance_df)), 
                      value=min(15, len(importance_df)))
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
    **Interpretation:** Features with higher importance scores have greater influence on the model's prediction. 
    This helps identify which customer attributes most strongly indicate deposit subscription likelihood.
    """)

except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())