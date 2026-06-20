"""
Bank Customer Deposit Prediction - Main Application

An end-to-end ML pipeline deployed as a multi-page Streamlit application.
Pages: Home (this), Single Prediction, Model Metrics, Feature Importance, Batch Prediction, EDA Dashboard
"""

import streamlit as st

# ----------------- PAGE CONFIG ----------------- #
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS ----------------- #
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .pipeline-step {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem;
    }
    .nav-card {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.25rem 0;
        background: #f9fafb;
        border-radius: 0.5rem;
        border-left: 3px solid #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER ----------------- #
st.markdown('<p class="main-header">💰 Bank Customer Deposit Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">End-to-End Machine Learning Pipeline for Customer Subscription Forecasting</p>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates a complete machine learning workflow — from data preprocessing 
to model deployment — for predicting whether a bank customer will subscribe to a term deposit.

**Business Context:** Banks run expensive direct marketing campaigns. This model identifies 
high-probability customers, reducing campaign costs and improving conversion rates.
""")

st.divider()

# ----------------- PIPELINE OVERVIEW ----------------- #
st.subheader("🔄 Pipeline Architecture")

col1, col2, col3, col4, col5 = st.columns(5)

pipeline_steps = [
    ("📥", "Data Ingestion", "#eff6ff", "#1e40af"),
    ("🧹", "Preprocessing", "#f0fdf4", "#166534"),
    ("⚙️", "Feature Engineering", "#fef3c7", "#92400e"),
    ("🤖", "Model Training", "#fdf2f8", "#9d174d"),
    ("🚀", "Deployment", "#f3e8ff", "#6b21a8"),
]

for col, (icon, title, bg_color, text_color) in zip([col1, col2, col3, col4, col5], pipeline_steps):
    with col:
        st.markdown(f"""
        <div class="pipeline-step" style="background: {bg_color};">
            <h3 style="margin: 0; color: {text_color};">{icon}</h3>
            <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: {text_color}; font-size: 0.85rem;">{title}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ----------------- KEY METRICS ----------------- #
st.subheader("📊 Project at a Glance")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(label="Model", value="Random Forest", delta="Ensemble | 100 Trees")
with col_m2:
    st.metric(label="Features", value="13", delta="3 Engineered")
with col_m3:
    st.metric(label="Dataset", value="11,162", delta="Records")
with col_m4:
    st.metric(label="Deployment", value="Live", delta="Streamlit Cloud")

st.divider()

# ----------------- NAVIGATION GUIDE ----------------- #
st.subheader("🧭 Application Pages")

pages_info = [
    ("🏠", "Home", "This page — project overview and navigation guide"),
    ("🎯", "Single Prediction", "Real-time customer scoring with probability gauge"),
    ("📊", "Model Metrics", "Confusion matrix, classification report, ROC-AUC analysis"),
    ("📈", "Feature Importance", "Random Forest feature rankings and interpretability"),
    ("📁", "Batch Prediction", "Upload CSV files to score multiple customers at once"),
    ("🔍", "EDA Dashboard", "Interactive exploration of distributions and relationships"),
]

for icon, title, description in pages_info:
    st.markdown(f"""
    <div class="nav-card">
        <span style="font-size: 1.5rem; margin-right: 1rem;">{icon}</span>
        <div>
            <p style="margin: 0; font-weight: 600; color: #1f2937;">{title}</p>
            <p style="margin: 0; font-size: 0.9rem; color: #6b7280;">{description}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ----------------- TECH STACK ----------------- #
st.subheader("🛠️ Tech Stack")

tech_stack = [
    ("Python 3.11", "Core language"),
    ("Pandas & NumPy", "Data manipulation"),
    ("Scikit-learn", "ML modeling"),
    ("Streamlit", "Web deployment"),
    ("Matplotlib & Seaborn", "Visualization"),
    ("Joblib", "Model serialization"),
    ("Git & GitHub", "Version control"),
    ("Jupyter", "Experimentation"),
]

tech_cols = st.columns(4)
for i, (tech, desc) in enumerate(tech_stack):
    with tech_cols[i % 4]:
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0; background: #f3f4f6; border-radius: 0.25rem;">
            <p style="margin: 0; font-weight: 600; font-size: 0.9rem; color: #1f2937;">{tech}</p>
            <p style="margin: 0; font-size: 0.75rem; color: #6b7280;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ----------------- FOOTER ----------------- #
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #9ca3af; font-size: 0.85rem;">
    <p>Built with focus on reproducibility, clarity, and real-world applicability.</p>
    <p>💻 <a href="https://github.com/MohanGC07/customer-purchase-prediction" target="_blank">View Source on GitHub</a></p>
</div>
""", unsafe_allow_html=True)