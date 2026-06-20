"""
Single Customer Prediction Page - Real-time scoring with probability output
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.predict import DepositPredictor

st.set_page_config(page_title="Single Prediction", page_icon="🎯")

st.title("🎯 Single Customer Prediction")

st.markdown("""
Enter customer details to receive an instant prediction on whether they will subscribe to a term deposit.
""")

st.divider()

# ----------------- INPUT FORM ----------------- #
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox(
        "Job",
        ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
         "retired", "self-employed", "services", "student", "technician",
         "unemployed", "unknown"]
    )
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    balance = st.number_input("Account Balance", value=1000, step=100)

with col2:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    campaign = st.number_input("Campaign Contacts", min_value=1, value=1)
    poutcome = st.selectbox("Previous Outcome", ["failure", "nonexistent", "success"])

# ----------------- PREDICTION ----------------- #
st.divider()

if st.button("Predict Subscription", type="primary", use_container_width=True):
    input_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "campaign": campaign,
        "poutcome": poutcome,
    }
    
    with st.spinner("Analyzing customer profile..."):
        predictor = DepositPredictor()
        prediction, probability = predictor.predict(input_data)
    
    # Result display
    if prediction == 1:
        st.success(f"✅ **Likely to Subscribe** — Probability: {probability:.1%}")
        st.balloons()
    else:
        st.warning(f"❌ **Unlikely to Subscribe** — Probability: {probability:.1%}")
    
    # Probability gauge
    st.progress(float(probability), text=f"Subscription Probability: {probability:.1%}")
    
    # Feature breakdown
    with st.expander("View Input Summary"):
        st.json(input_data)