# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------- LOAD MODEL & SCALER ----------------- #
rf_model = joblib.load("models/random_forest_model.pkl")  # Random Forest model
scaler = joblib.load("models/scaler.pkl")                 # StandardScaler
X_train_cols = joblib.load("models/X_train_columns.pkl")  # Columns from training

# ----------------- STREAMLIT CONFIG ----------------- #
st.set_page_config(page_title="Bank Deposit Prediction", page_icon="💰")
st.title("💰 Customer Deposit Subscription Prediction")
st.write("""
This app predicts whether a bank customer will subscribe to a **term deposit** based on their profile.
""")

# ----------------- USER INPUT ----------------- #
st.header("Customer Profile")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
job = st.selectbox("Job", ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'])
marital = st.selectbox("Marital Status", ['married','single','divorced'])
education = st.selectbox("Education Level", ['primary','secondary','tertiary','unknown'])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Housing Loan", ['yes','no'])
loan = st.selectbox("Personal Loan", ['yes','no'])
contact = st.selectbox("Contact Type", ['cellular','telephone'])
campaign = st.number_input("Number of Contacts", min_value=1, value=1)
poutcome = st.selectbox("Previous Outcome", ['failure','nonexistent','success'])

# ----------------- FEATURE ENGINEERING ----------------- #
# Age Group
if age <= 30:
    age_group = 'Young Adult'
elif age <= 45:
    age_group = 'Adult'
elif age <= 60:
    age_group = 'Middle-Aged'
else:
    age_group = 'Senior'

# Balance Category
if balance < 1000:
    balance_category = 'Low'
elif balance <= 5000:
    balance_category = 'Medium'
else:
    balance_category = 'High'

# Contact Intensity
if campaign <= 2:
    contact_intensity = 'Low'
elif campaign <= 5:
    contact_intensity = 'Medium'
else:
    contact_intensity = 'High'

# Build dataframe
input_dict = {
    'age': age,
    'balance': balance,
    'campaign': campaign,
    'job': job,
    'marital': marital,
    'education': education,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'poutcome': poutcome,
    'age_group': age_group,
    'balance_category': balance_category,
    'contact_intensity': contact_intensity
}

input_df = pd.DataFrame([input_dict])

# Encode categorical features
categorical_cols = ['job','marital','education','housing','loan','contact','poutcome','age_group','balance_category','contact_intensity']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=X_train_cols, fill_value=0)

# Scale features
input_scaled = scaler.transform(input_encoded)

# ----------------- PREDICTION ----------------- #
if st.button("Predict Deposit Subscription"):
    pred = rf_model.predict(input_scaled)[0]
    prob = rf_model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.success(f"✅ Customer is likely to subscribe to a term deposit! Probability: {prob:.2f}")
    else:
        st.warning(f"❌ Customer is unlikely to subscribe. Probability: {prob:.2f}")
