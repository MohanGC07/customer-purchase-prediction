# Bank Marketing - Customer Purchase Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

A comprehensive machine learning project that predicts whether a bank customer will subscribe to a term deposit based on demographic information, socio-economic data, and marketing campaign interactions.

---

## 👨‍🎓 Project Information

**Developer:** Mohan G.C  
**Training Program:** Data Science & Machine Learning (DSML) Course  
**Training Center:** Deerwalk Training Center  
**Instructor:** Sharad Sir  
**Duration:** 40 Days Training Program  

**Project Type:** Certification Capstone Project

---

## 📊 Project Overview

This capstone project demonstrates end-to-end machine learning workflow using the **Bank Marketing Dataset** from Kaggle. The project encompasses data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and deployment through an interactive web application.

The primary goal is to build predictive models that can accurately identify potential term deposit subscribers, enabling the bank to optimize their marketing campaigns and improve conversion rates.

**Dataset Source:** [Bank Marketing Dataset - Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

**Business Context:** Direct marketing campaigns (phone calls) by a Portuguese banking institution to promote term deposit subscriptions.

---

## 🎯 Project Objectives

- ✅ Perform comprehensive data cleaning and preprocessing
- ✅ Conduct in-depth exploratory data analysis (EDA)
- ✅ Engineer meaningful features for improved model performance
- ✅ Build and compare multiple classification algorithms
- ✅ Select the best-performing model with proper justification
- ✅ Deploy an interactive prediction system
- ✅ Provide actionable business insights

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

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- Git (optional, for version control)

### Setup Instructions

1. **Clone or Download the Repository**
```bash
git clone <repository-url>
cd customer-purchase-prediction
```

2. **Create Virtual Environment**
```bash
python -m venv venv
```

3. **Activate Virtual Environment**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. **Install Required Dependencies**
```bash
pip install -r requirements.txt
```

5. **Download the Dataset**
- Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- Download the CSV file
- Place it in the `data/raw/` directory

6. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

---

## 📦 Required Libraries

```txt
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

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running Jupyter Notebooks

Execute the notebooks in sequential order for complete analysis:

```bash
jupyter notebook
```

**Recommended Order:**

1. **01_data_loading.ipynb**
   - Load dataset from Kaggle
   - Display shape, info(), describe()
   - Initial data inspection

2. **02_data_cleaning.ipynb**
   - Handle missing/unknown values
   - Outlier detection and treatment
   - Data type corrections

3. **03_eda.ipynb**
   - Univariate analysis (distributions)
   - Bivariate analysis (relationships)
   - Generate insights and visualizations

4. **04_feature_engineering.ipynb**
   - Create age groups
   - Balance categories
   - Contact intensity metrics

5. **05_modeling.ipynb**
   - Train multiple ML models
   - Evaluate and compare performance
   - Select best model
   - Make predictions

### Running Streamlit Application

Launch the interactive web app for real-time predictions:

```bash
streamlit run app.py
```

**App Features:**
- Input customer information through user-friendly interface
- Get instant subscription predictions
- View model confidence scores
- Explore feature importance
- Access model performance metrics

---

## 📈 Project Workflow & Evaluation Criteria

### 1. Data Collection (5 marks)
- ✅ Dataset downloaded and loaded successfully
- ✅ Displayed first 10 rows
- ✅ Dataset shape and structure documented
- ✅ Statistical summaries generated

### 2. Data Cleaning & Transformation (20 marks)

**2.1 Missing/Unknown Values**
- Identified "unknown" entries across categorical columns
- Applied appropriate imputation strategies
- Documented decision rationale

**2.2 Outlier Detection & Treatment**
- Used boxplots and IQR method
- Analyzed: age, balance, campaign
- Applied capping/removal with justification

**2.3 Data Type Corrections**
- Ensured categorical variables properly typed
- Converted binary yes/no columns
- Validated numerical columns

**2.4 Feature Engineering**
Created meaningful features:
- **Age Group:** Young Adult, Adult, Middle-Aged, Senior
- **Balance Category:** Low, Medium, High
- **Contact Intensity:** Low, Medium, High campaigns

### 3. Exploratory Data Analysis (25 marks)

**3.1 Univariate Analysis**
- Age distribution
- Balance distribution
- Job type frequencies
- Target variable distribution

**3.2 Bivariate Analysis**
- Balance vs. Subscription rate
- Age group vs. Subscription
- Campaign frequency impact
- Job type subscription patterns
- Education level influence

**3.3 Key Insights**
Generated 5+ actionable business insights

### 4. Machine Learning Modeling (30 marks)

**Models Implemented:**
Choose any 3 from:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

**Evaluation Metrics:**
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-Score
- Classification Report

### 5. Best Model Selection (10 marks)
Comprehensive comparison based on:
- Performance metrics
- Overfitting/underfitting analysis
- Model interpretability
- Business applicability

### 6. Model Prediction (10 marks)
- Test set predictions (first 20 samples)
- Custom user input predictions
- Clear result interpretation

### 7. Conclusion (5 marks)
- Data quality assessment
- Key trends summary
- Best model justification
- Limitations identified
- Future recommendations

---

## 🎁 Bonus Features (Extra Credit)

- [ ] **SMOTE Implementation** - Handle class imbalance
- [ ] **Cross-Validation** - K-fold validation for robust evaluation
- [ ] **ROC-AUC Curve** - Visual performance assessment
- [ ] **Feature Importance** - Analysis for tree-based models
- [ ] **Hyperparameter Tuning** - GridSearchCV or RandomizedSearchCV
- [ ] **Model Deployment** - Streamlit application

---

## 📊 Expected Insights

Key business insights to discover during EDA:

1. **Customer Demographics**
   - Which age groups show higher subscription rates?
   - How does marital status affect decision-making?

2. **Financial Indicators**
   - Does account balance correlate with subscription likelihood?
   - Impact of existing loans on term deposit interest

3. **Campaign Effectiveness**
   - Optimal number of contact attempts
   - Best communication channels (cellular vs. telephone)
   - Impact of previous campaign outcomes

4. **Socio-Economic Factors**
   - Education level influence
   - Job type patterns
   - Seasonal trends

5. **Risk Segmentation**
   - High-potential customer profiles
   - Low-conversion segments to avoid

---

## 🎯 Model Performance Comparison

*(Results to be updated after model training)*

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | - | - | - | - | - |
| KNN | - | - | - | - | - |
| Decision Tree | - | - | - | - | - |
| Random Forest | - | - | - | - | - |

**Best Model:** *[To be determined after evaluation]*

**Justification:** *[Based on metrics, interpretability, and business requirements]*

---

## 💡 Key Learnings & Takeaways

This project demonstrates proficiency in:

- **Data Wrangling:** Handling real-world messy data
- **Statistical Analysis:** Understanding data distributions and relationships
- **Feature Engineering:** Creating meaningful predictive variables
- **Machine Learning:** Implementing and comparing multiple algorithms
- **Model Evaluation:** Using appropriate metrics for imbalanced classification
- **Business Communication:** Translating technical findings to actionable insights
- **Deployment:** Building user-friendly prediction interfaces

---

## 🔮 Future Enhancements

Potential improvements for production deployment:

1. **Advanced Models**
   - Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - Neural Networks with TensorFlow/PyTorch
   - Ensemble methods (Stacking, Blending)

2. **Feature Engineering**
   - Time-based features (seasonality, day of week)
   - Interaction terms between features
   - Polynomial features for non-linear relationships

3. **Deployment**
   - REST API using Flask/FastAPI
   - Cloud deployment (AWS, Azure, GCP)
   - Docker containerization
   - CI/CD pipeline

4. **Monitoring**
   - Model drift detection
   - Performance tracking
   - A/B testing framework
   - Automated retraining pipeline

5. **Business Integration**
   - CRM system integration
   - Real-time scoring system
   - Automated campaign recommendations
   - ROI tracking dashboard

---

## 🛠️ Troubleshooting

**Common Issues:**

1. **Module Import Error**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Jupyter Kernel Issues**
   ```bash
   python -m ipykernel install --user --name=venv
   ```

3. **Dataset Not Found**
   - Ensure CSV file is in `data/raw/` directory
   - Check file name matches code

4. **Memory Issues**
   - Close unnecessary applications
   - Use data sampling for initial exploration
   - Consider using chunks for large datasets

---

## 📚 References & Resources

**Dataset:**
- [Bank Marketing Dataset - Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Documentation:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

**Research Paper:**
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.

---

## 👤 Author

**Mohan G.C**  
Data Science & Machine Learning Trainee  
Deerwalk Training Center

*This project was developed as part of the DSML certification program under the guidance of Sharad Sir.*

---

## 🙏 Acknowledgments

- **Sharad Sir** - Course Instructor 
- **Deerwalk Training Center** - For providing comprehensive DSML training
- **Kaggle Community** - For dataset and resources
- **UCI Machine Learning Repository** - Original data source
- **Banking Institution** - For making this dataset available for research

---

## 📝 License

This project is developed for **educational and certification purposes** as part of the Data Science & Machine Learning training program at Deerwalk Training Center.

**Usage Guidelines:**
- ✅ Educational use and learning
- ✅ Portfolio demonstration
- ✅ Reference for similar projects
- ❌ Commercial use without permission
- ❌ Claiming as original work without attribution

---

## 📧 Contact & Feedback

For questions, suggestions, or collaboration:

- **Email:** *[Your email address]*
- **LinkedIn:** *[Your LinkedIn profile]*
- **GitHub:** *[Your GitHub profile]*

---

## 🌟 Project Status

**Current Status:** ✅ Completed / 🚧 In Progress

**Completion Date:** *[Date]*

**Certification Status:** Pending Evaluation

---

**⭐ If you found this project helpful, please consider giving it a star!**


---

*Last Updated: January 2026*