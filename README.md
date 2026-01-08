# 💰 Bank Marketing – Customer Deposit Subscription Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/)

An end-to-end **Machine Learning classification project** that predicts whether a bank customer will subscribe to a **term deposit**, based on customer demographics, financial information, and marketing campaign interactions.  
The final solution is **trained, evaluated, and deployed as a Streamlit web application** for real-time prediction.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-purchase-prediction-mohan.streamlit.app/)

---

## 🌐 Live Application & Repository

- 🔗 **Streamlit App:**  
  https://customer-purchase-prediction-mohan.streamlit.app/

- 🔗 **GitHub Repository:**  
  https://github.com/MohanGC07/customer-purchase-prediction


---

## 👨‍🎓 Project Information

- **Developer:** Mohan G.C  
- **Program:** Data Science & Machine Learning (DSML)  
- **Training Center:** Deerwalk Training Center  
- **Instructor:** Sharad Sir  
- **Project Type:** Certification Capstone Project  

---

## 📊 Project Overview

Banks invest heavily in direct marketing campaigns to promote term deposits. However, contacting every customer is inefficient, costly, and often ineffective.

This project applies **machine learning** to:
- Analyze historical customer data
- Identify patterns influencing subscription behavior
- Predict the likelihood of a customer subscribing to a term deposit

The project follows a **complete machine learning lifecycle**:

**Data Preprocessing → Feature Engineering → EDA → Model Training → Evaluation → Deployment**

### 📂 Dataset Source
- **Bank Marketing Dataset (Kaggle)**  
  https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset  
- Original source: UCI Machine Learning Repository

---

## 🎯 Project Objectives

- Clean and preprocess real-world banking data  
- Perform exploratory data analysis (EDA)  
- Engineer meaningful features to improve prediction accuracy  
- Train and compare multiple classification models  
- Select the best-performing model using evaluation metrics  
- Deploy the final model using Streamlit  

---

## 🧠 Machine Learning Models Used

The following models were implemented and evaluated:

- Logistic Regression  
- Decision Tree Classifier  
- **Random Forest Classifier** ⭐ *(Final Selected Model)*  

### 📌 Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-Score  

---

## 🧩 Feature Engineering

To improve predictive performance, the following features were engineered:

### 🔹 Age Group
- Young Adult  
- Adult  
- Middle-Aged  
- Senior  

### 🔹 Balance Category
- Low  
- Medium  
- High  

### 🔹 Contact Intensity
- Low  
- Medium  
- High (based on number of campaign contacts)

Categorical features were encoded using **One-Hot Encoding**, and numerical features were standardized using **StandardScaler**.

---

## 📁 Project Structure
```
customer-purchase-prediction/
│
├── data/
│   ├── raw/                      # Original Kaggle dataset
│   └── processed/                # Cleaned and transformed data
│
├── notebooks/
│   ├── 01_data_loading.ipynb            # Initial data exploration
│   ├── 02_data_cleaning.ipynb           # Data preprocessing
│   ├── 03_eda.ipynb                     # Exploratory analysis
│   ├── 04_feature_engineering.ipynb     # Feature creation
│   └── 05_modeling.ipynb                # Model training & evaluation
│
├── models/
│   └── scaler.pkl                # Saved preprocessing objects
│
├── venv/                         # Virtual environment
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore configuration

```


---

## 🔧 Installation & Setup

**Prerequisites**
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- Git (optional, for version control)

**Setup Instructions**
**Clone or Download the Repository**
```
git clone <repository-url>
cd customer-purchase-prediction

```

**Create Virtual Environment**
```
python -m venv venv

```
**Activate Virtual Environment**
**Windows:**
```
venv\Scripts\activate

```

**macOS/Linux:**
```
source venv/bin/activate

```

**Install Required Dependencies**
```
pip install -r requirements.txt

```
**Download the Dataset**
```
- Visit Kaggle Dataset Page
- Download the CSV file
- Place it in the data/raw/ directory
- Launch Jupyter Notebook
- jupyter notebook
```
**📦 Required Libraries**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
streamlit>=1.25.0
plotly>=5.14.0
imbalanced-learn>=0.10.0  # For SMOTE (bonus feature)

```
**Install all at once:**
```
pip install -r requirements.txt

```

**Run Streamlit App**
```
streamlit run app.py
```

---

## 🚀 Streamlit Application

**The deployed app allows users to:**
- Enter customer details
- Predict term deposit subscription
- View prediction probability
- Use the trained Random Forest model in real time
- The app loads:
**Trained model** :(random_forest_model.pkl)
**Scaler** :(scaler.pkl)
**Training feature columns** :(X_train_columns.pkl)
- This ensures consistent preprocessing during inference.

---

## 📈 Model Comparison (Accuracy)
```
Model	                                        Accuracy
Logistic Regression	                         Evaluated
Decision Tree	                               Evaluated
Random Forest	                               ⭐ Best

- Final Model Selected: Random Forest Classifier
- Reason: Better accuracy, robustness, and generalization performance.

```

---

## 💾 Model Persistence

To ensure reusability and deployment readiness, the following were saved using joblib:
- Trained Random Forest model
- StandardScaler
- Training feature column order
This prevents feature mismatch and scaling errors during prediction.

---

## 🔍 Key Learnings

- Real-world datasets require careful cleaning and validation
- Feature engineering significantly improves model performance
- Ensemble models like Random Forest provide better stability
- Model deployment requires consistent preprocessing pipelines
- Streamlit enables fast and effective ML deployment

---

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearchCV)
- ROC-AUC analysis
- Feature importance visualization
- REST API deployment (FastAPI)
- Model monitoring and retraining

---

👤 Author
```

Mohan G.C
Computer Engineering Student
Data Science & Machine Learning Trainee
Deerwalk Training Center

```

---

📜 License
```

This project is intended for educational and academic purposes.
Free to use for learning, projects, and portfolio demonstration with proper attribution.

⭐ If this project helped you, feel free to star the repository.

```

---