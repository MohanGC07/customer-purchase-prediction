"""
Exploratory Data Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import RAW_DATA_PATH

st.set_page_config(page_title="EDA Dashboard", page_icon="🔍")

st.title("🔍 Exploratory Data Analysis")

try:
    df = pd.read_csv(RAW_DATA_PATH)
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    # Column selector
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Categorical Analysis"])
    
    with tab1:
        st.subheader("Numeric Feature Distributions")
        selected_num = st.selectbox("Select numeric feature", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_num], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribution of {selected_num}")
        st.pyplot(fig)
        
        st.dataframe(df[selected_num].describe().to_frame().T)
    
    with tab2:
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) > 1:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("Not enough numeric columns for correlation analysis.")
    
    with tab3:
        st.subheader("Categorical Feature Analysis")
        selected_cat = st.selectbox("Select categorical feature", categorical_cols)
        
        value_counts = df[selected_cat].value_counts().head(10)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax3, palette="Set2")
        ax3.set_title(f"Top Categories in {selected_cat}")
        st.pyplot(fig3)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure your raw data file exists at data/raw/bank.csv")
    