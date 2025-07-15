# Titanic Survival Prediction

Predict whether a passenger survived the Titanic disaster using an end‑to‑end Python + scikit‑learn workflow. This project covers data cleaning, feature engineering, preprocessing pipelines, model training, evaluation, and submission generation for Kaggle.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Environment & Installation](#environment--installation)  
4. [Data Cleaning & Imputation](#data-cleaning--imputation)  
5. [Feature Engineering](#feature-engineering)  
6. [Preprocessing Pipeline](#preprocessing-pipeline)  
7. [Modeling](#modeling)  
8. [Evaluation](#evaluation)  
9. [Submission](#submission)  
10. [Results](#results)  
11. [License](#license)  

---

## 🔎 Project Overview

Build a machine learning model to predict Titanic passenger survival based on features such as:

- **Pclass** (ticket class)  
- **Sex**  
- **Age**  
- **SibSp** (# siblings/spouses aboard)  
- **Parch** (# parents/children aboard)  
- **Fare**  
- **Embarked** (port of embarkation)  
- Engineered features: family size, honorific titles, age‑bucket flags  

Kaggle competition: https://www.kaggle.com/c/titanic

---

## 📁 Repository Structure

```
├── .gitignore
├── README.md             # this file
├── requirements.txt      # Python dependencies
├── train.csv             # training data with labels
├── test.csv              # test data without labels
├── titanic.ipynb         # Jupyter notebook: end-to-end pipeline
├── model.ipynb           # standalone modeling notebook
└── submission.csv        # example submission file
```

---

## ⚙️ Environment & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate.bat   # Windows (cmd)
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Key dependencies:** numpy, pandas, scikit-learn, matplotlib, seaborn

---

## 🧹 Data Cleaning & Imputation

Missing values are handled by a custom Imputer transformer:

- **Age**: filled with the training-set median  
- **Embarked**: filled with the most frequent port per passenger class  
- **Fare**: if missing, filled with the mean fare per passenger class  

Usage:

```python
imputer = Imputer(age_strategy='median')
train_clean = imputer.fit_transform(train_df)
test_clean  = imputer.transform(test_df)
```

---

## 🔧 Feature Engineering

1. **Title extraction** from Name  
2. **Title grouping** into broader categories (Military, Noble, Miss, Mrs, Mr, Master, Other)  
3. **FamilySize** = SibSp + Parch + 1  
4. **Age-bucket flags**:  
   - is_child (<18)  
   - is_young_adult (18–35)  
   - is_adult (35–60)  
   - is_senior (>=60)  

---

## 🛠️ Preprocessing Pipeline

Chain all steps into an sklearn Pipeline:

```python
full_pipeline = Pipeline([
    ('impute', Imputer(age_strategy='median')),
    ('title', FunctionTransformer(initialize_title, validate=False)),
    ('tgroup', FunctionTransformer(initialize_title_group, validate=False)),
    ('family', FunctionTransformer(initialize_family_size, validate=False)),
    ('ageflags', FunctionTransformer(initialize_age_flags, validate=False)),
    ('encode_sex', FunctionTransformer(encode_sex, validate=False)),
    ('encode_emb', FunctionTransformer(encode_embarked, validate=False)),
    ('encode_tg', FunctionTransformer(encode_title_group, validate=False)),
    ('select', FunctionTransformer(drop_unnecessary_columns, validate=False))
])

X_train = full_pipeline.fit_transform(train_df)
X_test  = full_pipeline.transform(test_df)
y_train = train_df['Survived']
```

---

## 🤖 Modeling

Example with Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
```

---

## 📊 Evaluation

- Hold-out validation (train/validation split)  
- 5-fold cross-validation  
- Metrics: accuracy, precision, recall, F1  
- Confusion matrix & classification report  

```python
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
```

---

## 🏁 Submission

```python
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': rf.predict(X_test).astype(int)
})
submission.to_csv('submission.csv', index=False)
```

---

## 📈 Results

- Public leaderboard score: 0.77033  
- Top features: Sex, Fare, Title_Group, FamilySize, Pclass  

---

## 📝 License

MIT License
