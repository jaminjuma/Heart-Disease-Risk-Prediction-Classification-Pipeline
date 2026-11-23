# ❤️ Heart Disease Risk Prediction — Classification Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/imbalanced--learn-SMOTE-teal" alt="imblearn">
  <img src="https://img.shields.io/badge/Seaborn-Visualization-4C72B0" alt="Seaborn">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  A machine learning pipeline that predicts whether a patient has <b>heart disease</b> from
  demographic, medical, and lifestyle data — with leakage-safe preprocessing, SMOTE for class
  imbalance, and tuned models compared on ROC-AUC.
</p>

---

## 📖 Overview

This project builds a classifier to predict `has_heart_disease` (0 = No, 1 = Yes) from patient
demographics, medical test results, and lifestyle habits. The primary goal is to **balance precision
and recall**, so heart disease risk is detected early without too many false alarms.

The notebook covers:

- 🔍 **Exploration** — distributions, skewness, outliers, and group comparisons
- ⚙️ **Preprocessing** — a modular `ColumnTransformer` pipeline
- ⚖️ **Imbalance handling** — SMOTE applied inside cross-validation folds
- 🤖 **Modeling** — four algorithms tuned with `GridSearchCV`
- 📈 **Evaluation** — confusion matrix, ROC curve, and full classification metrics

---

## 🗂️ Dataset

| Detail | Description |
|---|---|
| **Source** | <dataset source / link> |
| **File** | `<filename>.csv` |
| **Size** | <rows × columns> |
| **Target** | `has_heart_disease` (binary) |
| **Continuous features** | `age`, resting diastolic blood pressure, `st_depression`, `alcohol_units_per_week`, `max_heart_rate_achieved` |
| **Categorical features** | `sex`, smoking status, <others> |

> 📥 The raw data is not included in this repo. Place the file in the working directory before running
> the notebook.

---

## 📊 Exploratory Analysis & Key Insights

- **Skewed distributions:** `st_depression` and `alcohol_units_per_week` are right-skewed, while
  `max_heart_rate_achieved` is slightly left-skewed. Outlier analysis found very few extreme values, so
  outliers are not a substantial issue.
- **Alcohol vs. sex & smoking:** Males drink more per week on average. Male smokers drink the most,
  while female never-smokers drink more than female current smokers.
- **Age trends:** Blood pressure rises gradually with age, and heart disease cases are concentrated among
  older individuals with higher diastolic blood pressure.

---

## ⚙️ Data Preprocessing Pipeline

A modular `ColumnTransformer` keeps all transformations inside the pipeline to **prevent data leakage**:

| Step | Applied to | Method |
|---|---|---|
| **Scaling** | Numerical features | `StandardScaler` |
| **Encoding** | Categorical features | `OneHotEncoder` |
| **Class imbalance** | Training folds only | `SMOTE` inside an `imblearn` Pipeline |

---

## 🤖 Model Exploration & Tuning

Models were compared with **5-fold `StratifiedKFold`** cross-validation, tuned with **`GridSearchCV`**
on **ROC-AUC**:

| Model | Best parameters | CV ROC-AUC |
|---|---|---|
| **Logistic Regression** | <params> | <score> |
| **Support Vector Machine** | <params> | <score> |
| **Random Forest** | <params> | <score> |
| **Gradient Boosting** | <params> | <score> |

---

## 🏆 Final Model Performance

The best estimator is evaluated on an untouched **20% test set**.

**Selected model:** <model name>

| Metric | Score |
|---|---|
| Accuracy | <0.00> |
| Precision | <0.00> |
| Recall | <0.00> |
| F1-Score | <0.00> |
| ROC-AUC | <0.00> |

Visualizations include a **confusion matrix** (true vs. false positives) and an **ROC curve**
(discriminative ability).

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Modeling** — scikit-learn, imbalanced-learn
- **Visualization** — matplotlib, seaborn
- **Environment** — developed in Google Colab / Jupyter

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/jaminjuma/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Place the dataset in the project folder, then run the notebook
jupyter notebook
```

---

## 🔮 Possible Next Steps

- Tune the decision threshold to favor recall for early detection
- Try XGBoost / LightGBM and compare against the current best model
- Add SHAP explanations to show which features drive each prediction
- Probability calibration for more reliable risk scores
- Deploy as a simple web app for risk screening

---

## 👤 Author

**Jamin Juma**
Founder of [Automateke](https://automateke.app) · CS student, Kirinyaga University

- 🌐 Portfolio: [jaminjuma.tech](https://jaminjuma.tech)
- 💻 GitHub: [@jaminjuma](https://github.com/jaminjuma)
- 🔗 LinkedIn: [jumajamin](https://linkedin.com/in/jumajamin)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and adapt it.
