"""
Batch Prediction Page - Upload CSV and score multiple customers
"""

import streamlit as st
import pandas as pd
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.predict import DepositPredictor

st.set_page_config(page_title="Batch Prediction", page_icon="📁")

st.title("📁 Batch Prediction")

st.markdown("""
Upload a CSV file with customer data to get predictions for multiple customers at once.

**Required columns:** `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, `contact`, `campaign`, `poutcome`
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records")
        
        st.subheader("Preview")
        st.dataframe(df.head(10))
        
        required_cols = ["age", "job", "marital", "education", "balance", 
                        "housing", "loan", "contact", "campaign", "poutcome"]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    predictor = DepositPredictor()
                    
                    results = []
                    for idx, row in df.iterrows():
                        input_dict = {
                            "age": row["age"],
                            "job": row["job"],
                            "marital": row["marital"],
                            "education": row["education"],
                            "balance": row["balance"],
                            "housing": row["housing"],
                            "loan": row["loan"],
                            "contact": row["contact"],
                            "campaign": row["campaign"],
                            "poutcome": row["poutcome"],
                        }
                        pred, prob = predictor.predict(input_dict)
                        results.append({
                            "Prediction": "Deposit" if pred == 1 else "No Deposit",
                            "Probability": prob,
                            "Will_Subscribe": pred
                        })
                    
                    results_df = pd.DataFrame(results)
                    output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                    
                    st.subheader("Results")
                    st.dataframe(output_df)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    total = len(results_df)
                    deposits = results_df["Will_Subscribe"].sum()
                    col1.metric("Total Customers", total)
                    col2.metric("Likely Subscribers", deposits)
                    col3.metric("Conversion Rate", f"{deposits/total:.1%}")
                    
                    # Download button
                    csv = output_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to begin batch prediction.")
    
    # Template download
    template = pd.DataFrame(columns=["age", "job", "marital", "education", "balance", 
                                     "housing", "loan", "contact", "campaign", "poutcome"])
    template_csv = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV Template",
        data=template_csv,
        file_name="batch_template.csv",
        mime="text/csv"
    )