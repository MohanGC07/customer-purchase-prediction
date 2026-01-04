# Customer Purchase Prediction

End-to-end Data Science & Machine Learning project using the Bank Marketing Dataset.

## Objectives
- Predict whether a customer will subscribe to a term deposit
- Perform data cleaning, EDA, feature engineering
- Train and compare multiple ML models
- Deploy the best model using Streamlit

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit


## 📊 Dataset Collection & Initial Inspection
## Dataset Source

The dataset used in this project is the Bank Marketing Dataset, obtained from Kaggle. It contains information related to direct marketing campaigns conducted by a banking institution, with the objective of predicting whether a customer will subscribe to a term deposit.

- Dataset Name: Bank Marketing Dataset
- Source: Kaggle
- Target Variable: y (Yes / No – term deposit subscription)

## Dataset Loading

The dataset was downloaded manually from Kaggle and stored in the data/raw/ directory to preserve the original data. It was loaded into a Jupyter Notebook using the pandas library for analysis.

## Initial Data Inspection

To understand the structure and quality of the data, the following exploratory steps were performed:

- First 10 Rows: Displayed to observe sample records and understand feature values.
- Dataset Shape: Examined to determine the total number of observations and features.
- Dataset Information (.info()): Used to inspect data types, non-null counts, and identify categorical and numerical columns.
- Statistical Summary (.describe()): Generated for numerical features to analyze central tendency, dispersion, and potential outliers.

## Key Observations

- The dataset contains a mix of categorical and numerical features.
- Several categorical columns include "unknown" values, which require careful handling during data cleaning.
- Numerical features such as age, balance, and campaign count show varying ranges, indicating the need for outlier analysis and potential transformation.
- This initial inspection provides the foundation for subsequent steps, including data cleaning, feature engineering, exploratory data analysis, and machine learning modeling.

## 🔧 Project Structure (Relevant Files)
notebooks/
 └── 01_data_loading.ipynb   # Dataset loading and initial inspection
data/
 └── raw/
     └── bank.csv            # Original dataset from Kaggle
