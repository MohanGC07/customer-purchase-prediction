"""
Bank Customer Deposit Prediction - Main Application
"""

import streamlit as st

st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- HEADER ----------------- #
st.title("💰 Bank Customer Deposit Prediction")
st.markdown("""
Welcome to the **Bank Customer Deposit Prediction** platform.

This application demonstrates an end-to-end machine learning pipeline:
- **Real-time Prediction** — Single customer scoring
- **Model Metrics** — Performance evaluation and ROC analysis
- **Feature Importance** — Understand what drives subscription behavior
- **Batch Prediction** — Score thousands of customers via CSV upload
- **EDA Dashboard** — Interactive data exploration

Use the sidebar to navigate between sections.
""")

st.divider()

# Quick stats cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Model Type", value="Random Forest")
with col2:
    st.metric(label="Features", value="13 Engineered")
with col3:
    st.metric(label="Deployment", value="Streamlit Cloud")

st.info("👈 Select a page from the sidebar to get started.")