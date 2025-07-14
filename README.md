# Titanic Survival Prediction 

This repository contains a complete end-to-end data science workflow to predict passenger survival on the Titanic using Python and scikit-learn. The process includes data cleaning, feature engineering, preprocessing pipelines, model training, evaluation, and submission generation for Kaggle.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment & Installation](#environment--installation)
4. [Data Cleaning & Imputation](#data-cleaning--imputation)
5. [Feature Engineering](#feature-engineering)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Modeling](#modeling)
8. [Evaluation](#evaluation)
9. [Submission](#submission)
10. [License](#license)

---

## Project Overview

The goal is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on features like class, sex, age, fare, and engineered attributes (family size, titles, age buckets).

Kaggle leaderboard: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

---

## Repository Structure

```
├── train.csv           # training data with ground truth
├── test.csv            # test data without labels
├── titanic.ipynb       # Jupyter notebook with all steps
├── model.ipynb         # standalone modeling notebook
├── submission.csv      # example submission
└── README.md           # this documentation
```

---

## Environment & Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

**Key dependencies:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Data Cleaning & Imputation

* **Imputer** class fills missing values:

  * `Age`: median by default (or mean option)
  * `Embarked`: most frequent port per passenger class
  * `Fare`: mean fare per passenger class

```python
imputer = Imputer(age_strategy='median')
train_df = imputer.fit_transform(train_df)
test_df  = imputer.transform(test_df)
```

---

## Feature Engineering

1. **Title extraction** from `Name` (e.g., `Mr`, `Miss`, `Master`).
2. **Title grouping** into categories: Military, Noble, Miss, Mrs, Mr, Master, Other.
3. **FamilySize** = `SibSp + Parch + 1`.
4. **Age buckets** flags (`is_child`, `is_young_adult`, `is_adult`, `is_senior`).

---

## Preprocessing Pipeline

We chain all steps into an sklearn `Pipeline` using `FunctionTransformer`:

```python
full_pipeline = Pipeline([
    ('impute', Imputer(age_strategy='median')),
    ('title',  FunctionTransformer(initialize_title)),
    ('family', FunctionTransformer(initialize_family_size)),
    ('ageflags', FunctionTransformer(initialize_age_flags)),
    ('tgroup', FunctionTransformer(initialize_title_group)),
    ('sex',   FunctionTransformer(encode_sex)),
    ('emb',   FunctionTransformer(encode_embarked)),
    ('tgrp_enc', FunctionTransformer(encode_title_group)),
    ('select', FunctionTransformer(drop_unnecessary_columns))
])

X_train = full_pipeline.fit_transform(train_raw)
X_test  = full_pipeline.transform(test_raw)
y_train = train_raw['Survived']
```

---

## Modeling

Example using Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
```

Other models (logistic regression, XGBoost) can be swapped in and cross-validated similarly.

---

## Evaluation

* **Train/Validation split:** accuracy, precision, recall, F1, confusion matrix.
* **Cross-validation:** 5-fold on full training data.
* **Feature importance:** for tree-based models.

```python
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
```

---

## Submission

Generate a CSV for Kaggle:

```python
submission = pd.DataFrame({
    'PassengerId': test_raw['PassengerId'],
    'Survived':    rf.predict(X_test).astype(int)
})
submission.to_csv('submission.csv', index=False)
```

Submit `submission.csv` to the Titanic competition on Kaggle.

---

