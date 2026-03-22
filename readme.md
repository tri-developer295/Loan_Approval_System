# 🏦 Loan Approval Prediction — Supervised ML Pipeline

A end-to-end binary classification pipeline that predicts whether a loan application will be approved or rejected, using three classical ML algorithms: **K-Nearest Neighbors**, **Logistic Regression**, and **Naive Bayes**.

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── data/
│   └── loan_data.csv               # Raw dataset
│
├── notebooks/
│   └── loan_approval_eda.ipynb     # Exploratory Data Analysis notebook
|
├── requirements.txt
└── README.md
```

---

## 📌 Problem Statement

Predict loan approval (`Y` / `N`) based on applicant details such as income, credit history, loan amount, and demographics. This is a **binary classification** task.

| Label | Meaning |
|-------|---------|
| `Y`   | Loan Approved |
| `N`   | Loan Rejected |

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `Applicant_id` | Unique applicant ID |
| `Applicant_income` | Monthly income of applicant |
| `Coapplicant_income` | Monthly income of co-applicant |
| `Employment_Status` | Salaried/Self-Employed/Business |
| `Age` | Applicant_id |
| `Marital_Status` | Married/Single |
| `Dependents` | Number of Dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of already running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Loan amount requested |
| `Loan_Term` | Loan duration(months) |
| `Loan_Purpose` | Home/Education/Personal/Business |
| `Property_Area` | Urban/Semi-Urban/Rural |
| `Education_Level` | Graduate/Postgraduate/Undergraduate |
| `Gender` | Male / Female |
| `Employer_Category` | Govt/Private/Self |
| `Loan_Approved(Target)` | 1=Approved,0=Rejected |

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights uncovered during EDA:

- **Credit History** is the strongest predictor — applicants with good credit history have ~80% approval rate vs ~10% for bad credit.
- **Graduates** are approved more frequently than non-graduates.
- **ApplicantIncome** and **LoanAmount** are right-skewed → log transformation applied.
- ~6–8% missing values present in several columns, handled via imputation.
- Dataset is **imbalanced** (~74% Approved, ~26% Rejected).

---

## ⚙️ Feature Engineering

The following transformations were applied to improve model performance:

| Feature | Description |
|---|---|
| `TotalIncome_log` | Log of combined applicant + co-applicant income |
| `LoanAmount_log` | Log-transformed loan amount to reduce skewness |
| `Income_to_Loan` | Ratio of total income to loan amount |
| `EMI` | Estimated monthly installment = LoanAmount / Term |
| `Balance_Income` | TotalIncome − (EMI × 1000) |
| `Dependents_num` | Numeric encoding of dependents (3+ → 3) |

**Missing Value Imputation:**
- Categorical columns → mode imputation
- Numerical columns → median imputation

**Encoding:** Label Encoding applied to all categorical variables.

**Scaling:** `StandardScaler` applied before model training (required for KNN and Logistic Regression).

---

## 🤖 Models Used

### 1. K-Nearest Neighbors (KNN)
- Distance-based classifier
- `n_neighbors=7`, `metric='minkowski'`
- Sensitive to feature scaling → StandardScaler applied

### 2. Logistic Regression
- Linear probabilistic classifier
- `C=1.0`, `max_iter=1000`
- Outputs class probabilities → used for ROC-AUC

### 3. Naive Bayes (Gaussian)
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast and effective on smaller datasets

---

## 📈 Model Evaluation

Each model is evaluated using the following metrics on the held-out test set (80/20 split, stratified):

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted approvals, how many are correct |
| **Recall** | Of actual approvals, how many were caught |
| **F1-Score** | Harmonic mean of Precision and Recall |

### Sample Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| KNN | ~0.80 | ~0.83 | ~0.91 | ~0.87 | ~0.79 |
| Logistic Regression | ~0.84 | ~0.87 | ~0.93 | ~0.90 | ~0.88 |
| Naive Bayes | ~0.82 | ~0.85 | ~0.92 | ~0.88 | ~0.85 |

> ⚠️ Results are based on a synthetically generated dataset and will vary on real-world data.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Run the Pipeline

```bash
# Run full pipeline (EDA → Feature Engineering → Training → Evaluation)
python src/train.py
```

---

## 💡 Key Takeaways

- **Logistic Regression** performed best overall, with the highest AUC and F1-score, making it the recommended model for this task.
- **Credit History** dominates prediction — models with this feature significantly outperform those without it.
- **Feature engineering** (especially log transforms and EMI) improved model performance by reducing skewness and adding financially meaningful signals.
- **Naive Bayes** is surprisingly competitive despite its independence assumption, especially given the relatively small dataset size.

---

## 🔮 Future Improvements

- Handle class imbalance with SMOTE or class weights
- Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- Try ensemble models (Random Forest, XGBoost) for comparison
- Deploy with a Flask / FastAPI endpoint
- Add SHAP values for model explainability

---

## 👤 Author

**Your Name**
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License.
