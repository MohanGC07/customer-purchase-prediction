# Bank Customer Deposit Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployment-Live-brightgreen?logo=rocket&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</p>

<p align="center">
  <b>An end-to-end machine learning pipeline for predicting bank term deposit subscriptions, deployed as a real-time web application.</b>
</p>

<p align="center">
  <a href="https://customer-purchase-prediction-mohan.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
</p>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Live Demo](#live-demo)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Problem Statement

Banks invest heavily in direct marketing campaigns to promote term deposits. However, contacting every customer in the database is:

- **Cost-inefficient** — marketing resources are wasted on low-probability prospects
- **Customer-fatiguing** — excessive contact reduces brand trust and response rates
- **Operationally unsustainable** — manual campaign targeting does not scale

**Objective:** Build a predictive model that identifies customers most likely to subscribe to a term deposit, enabling data-driven campaign targeting and measurable ROI improvement.

---

## Solution Overview

This project delivers a **production-ready classification pipeline** that:

1. **Ingests** historical campaign data (demographics, financials, contact history)
2. **Engineers** domain-informed features (age groups, balance tiers, contact intensity)
3. **Trains** and evaluates multiple classification algorithms
4. **Deploys** the best-performing model via an interactive Streamlit application
5. **Serves** real-time predictions with probability scores for business decision-making

**Business Impact:** Prioritize high-likelihood customers → reduce campaign cost → improve conversion rate.

---

## Live Demo

| Platform | Link |
|----------|------|
| **Streamlit App** | [customer-purchase-prediction-mohan.streamlit.app](https://customer-purchase-prediction-mohan.streamlit.app/) |
| **Repository** | [github.com/MohanGC07/customer-purchase-prediction](https://github.com/MohanGC07/customer-purchase-prediction) |

---

## Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Language** | Python 3.11 |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Machine Learning** | scikit-learn, imbalanced-learn |
| **Model Persistence** | joblib |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## Project Structure

```
customer-purchase-prediction/
│
├── data/
│   ├── raw/                          # Original dataset (Kaggle/UCI)
│   │   └── bank.csv
│   └── processed/                    # Cleaned & engineered datasets
│       ├── bank_cleaned.csv
│       └── bank_final.csv
│
├── notebooks/                        # Reproducible ML experiments
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda.ipynb
│   └── 05_modeling.ipynb
│
├── models/                           # Serialized artifacts for deployment
│   ├── random_forest_model.pkl       # Trained classifier
│   ├── scaler.pkl                    # Fitted StandardScaler
│   └── X_train_columns.pkl           # Training column order (prevents skew)
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

---

## ML Pipeline

```
Raw Data (bank.csv)
    ↓
Data Cleaning → Handle missing values, outliers, format inconsistencies
    ↓
Feature Engineering → age_group, balance_category, contact_intensity
    ↓
Exploratory Data Analysis → Distribution analysis, correlation, class imbalance
    ↓
Preprocessing → One-Hot Encoding, StandardScaler normalization
    ↓
Model Training → Logistic Regression, Decision Tree, Random Forest
    ↓
Evaluation → Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    ↓
Model Selection → Random Forest (best generalization)
    ↓
Serialization → joblib (model + scaler + column reference)
    ↓
Deployment → Streamlit Cloud (real-time inference)
```

### Feature Engineering

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `age_group` | Young Adult / Adult / Middle-Aged / Senior | Life-stage correlates with financial behavior |
| `balance_category` | Low / Medium / High | Account balance indicates investment capacity |
| `contact_intensity` | Low / Medium / High | Campaign fatigue affects conversion likelihood |

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| Logistic Regression | Evaluated | — | — | — | Baseline linear model |
| Decision Tree | Evaluated | — | — | — | Prone to overfitting |
| **Random Forest** | **Best** | **—** | **—** | **—** | **Selected: robust generalization, ensemble stability** |

> **Note:** Detailed metrics available in `notebooks/05_modeling.ipynb`.

### Why Random Forest?

- **Ensemble robustness** — reduces variance through bagging
- **Feature importance** — interpretable contribution scores
- **Non-linear patterns** — captures complex customer behavior
- **Stable generalization** — performs consistently on unseen data

---

## Key Features

### Application Capabilities

- **Real-time prediction** — input customer profile, get instant subscription probability
- **Probability scoring** — confidence-based output, not just binary classification
- **Consistent preprocessing** — serialized scaler and column alignment prevent training-serving skew
- **Interactive UI** — intuitive Streamlit interface for non-technical stakeholders

### Engineering Best Practices

- **Modular notebooks** — each stage (cleaning, EDA, modeling) is isolated and reproducible
- **Artifact versioning** — model, scaler, and column reference saved together
- **Deployment-ready** — all dependencies pinned in `requirements.txt`
- **Cloud-hosted** — accessible demo without local setup

---

## Installation

### Prerequisites

- Python 3.11+
- pip package manager
- Git (optional)

### Local Setup

```bash
# Clone repository
git clone https://github.com/MohanGC07/customer-purchase-prediction.git
cd customer-purchase-prediction

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scriptsctivate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
streamlit>=1.25.0
plotly>=5.14.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
jupyter>=1.0.0
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`.

### Run Notebooks

```bash
jupyter notebook
```

Open notebooks in sequence (`01` → `05`) to reproduce the full pipeline.

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Customer age (18–100) |
| `job` | Categorical | Occupation type |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `balance` | Numeric | Account balance |
| `housing` | Binary | Housing loan status |
| `loan` | Binary | Personal loan status |
| `contact` | Categorical | Contact communication type |
| `campaign` | Numeric | Number of contacts in current campaign |
| `poutcome` | Categorical | Outcome of previous campaign |

---

## Future Roadmap

- [ ] **Hyperparameter optimization** — GridSearchCV / Optuna for model tuning
- [ ] **ROC-AUC analysis** — threshold optimization for business cost sensitivity
- [ ] **Feature importance dashboard** — interactive SHAP/LIME explanations in Streamlit
- [ ] **Batch prediction API** — CSV upload for bulk scoring
- [ ] **REST API** — FastAPI backend for service integration
- [ ] **Model monitoring** — drift detection and automated retraining pipeline
- [ ] **A/B testing framework** — measure campaign lift from model-driven targeting

---

## License

This project is licensed under the MIT License — free for educational and portfolio use with attribution.

---

<p align="center">
  <i>Built with focus on reproducibility, clarity, and real-world applicability.</i>
</p>
