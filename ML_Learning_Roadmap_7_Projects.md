# 🎯 Machine Learning Mastery: 7 Portfolio-Worthy Projects

> **A Complete Learning Roadmap from Beginner to Expert**

This guide provides 7 carefully curated machine learning projects ordered from easy to difficult, designed to help you master ML fundamentals through hands-on feature engineering and model building.

---

## 📋 Table of Contents

1. [Learning Progression Overview](#learning-progression-overview)
2. [Essential Tools & Setup](#essential-tools--setup)
3. [Feature Engineering Best Practices](#feature-engineering-best-practices)
4. [Model Evaluation Metrics Guide](#model-evaluation-metrics-guide)
5. [Project 1: Titanic Survival Prediction](#project-1-titanic-survival-prediction-beginner)
6. [Project 2: House Price Prediction](#project-2-house-price-prediction-beginner-intermediate)
7. [Project 3: Customer Churn Prediction](#project-3-customer-churn-prediction-intermediate)
8. [Project 4: Credit Card Fraud Detection](#project-4-credit-card-fraud-detection-intermediate)
9. [Project 5: Store Sales Time Series Forecasting](#project-5-store-sales-time-series-forecasting-intermediate-advanced)
10. [Project 6: Sentiment Analysis on Reviews](#project-6-sentiment-analysis-on-reviews-intermediate-advanced)
11. [Project 7: Chest X-Ray Pneumonia Detection](#project-7-chest-x-ray-pneumonia-detection-advanced)
12. [Gradient Boosting Comparison Guide](#gradient-boosting-comparison-xgboost-vs-lightgbm-vs-catboost)
13. [Portfolio Building Tips](#portfolio-building-tips)
14. [Resources & References](#resources--references)

---

## Learning Progression Overview

```
Project 1-2 (Beginner)
├── Basic data cleaning & preprocessing
├── Simple feature engineering
├── Model comparison fundamentals
└── Cross-validation basics

Project 3-4 (Intermediate)
├── Class imbalance techniques
├── Advanced feature creation
├── Ensemble methods
└── Evaluation metric selection

Project 5-6 (Intermediate-Advanced)
├── Domain-specific features (time series, NLP)
├── Complex preprocessing pipelines
├── Deep learning integration
└── Hybrid model design

Project 7 (Advanced)
├── Computer vision techniques
├── Transfer learning mastery
├── Model explainability
└── Production-ready pipelines
```

---

## Essential Tools & Setup

### Required Libraries

```python
# Core Data Science Stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

# Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning (for advanced projects)
import tensorflow as tf
from tensorflow import keras
```

### Recommended Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install tensorflow  # For deep learning projects
pip install nltk spacy transformers  # For NLP projects
pip install opencv-python  # For image projects
```

---

## Feature Engineering Best Practices

Feature engineering is the **most impactful** step in any ML project. According to Kaggle Grandmasters, it often determines whether you win or lose a competition.

### Core Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Groupby Aggregations** | `df.groupby(COL1)[COL2].agg(['mean', 'std', 'count'])` | Tabular data with categorical columns |
| **Binning** | Convert continuous variables to categories | Age, income, price ranges |
| **Polynomial Features** | Create interaction terms (x1 * x2, x1²) | When relationships are non-linear |
| **Log Transformation** | `np.log1p(column)` | Skewed distributions |
| **Target Encoding** | Replace categories with target mean | High-cardinality categoricals |
| **Date Features** | Extract day, month, year, day_of_week | Time-based data |
| **Lag Features** | Previous values (t-1, t-7, t-30) | Time series data |
| **Rolling Statistics** | Rolling mean, std, min, max | Time series data |

### Standard Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column groups
numeric_features = ['age', 'income', 'balance']
categorical_features = ['gender', 'country', 'product_type']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

### Kaggle Grandmaster Tips

1. **Create thousands of features** - Let the model decide what works
2. **Treat every number as both numeric and categorical** - Sometimes binning helps
3. **Perfect cross-validation discipline** - Every model uses exactly the same folds
4. **Real model diversity** - Different features, preprocessing, seeds, and algorithms
5. **Groupby aggregations are king** - `groupby(COL1)[COL2].agg(STAT)` is incredibly powerful

---

## Model Evaluation Metrics Guide

### Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Balanced classes only |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Imbalanced data, need balance |
| **ROC-AUC** | Area under ROC curve | Binary classification, probability outputs |
| **PR-AUC** | Area under Precision-Recall curve | Highly imbalanced data (fraud detection) |
| **Log Loss** | -Σ(y·log(p) + (1-y)·log(1-p)) | Probability calibration matters |

### Regression Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **MSE** | Σ(y - ŷ)² / n | Penalize large errors heavily |
| **RMSE** | √MSE | Same units as target |
| **MAE** | Σ\|y - ŷ\| / n | Robust to outliers |
| **R²** | 1 - (SS_res / SS_tot) | Explain variance proportion |
| **MAPE** | Σ\|(y - ŷ) / y\| / n × 100 | Percentage error interpretation |

### Cross-Validation Template

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# For classification (maintains class distribution)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# For regression
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
print(f"RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Project 1: Titanic Survival Prediction (Beginner)

### Real-World Problem
Predict passenger survival based on demographics and ticket information — the foundational classification problem that every ML practitioner should master.

### Dataset
- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Size:** ~900 training samples
- **Features:** 11 columns including Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

### Feature Engineering Focus

```python
import pandas as pd
import numpy as np

def engineer_titanic_features(df):
    # 1. Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # 2. Create FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 4. Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                        'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 5. Age binning
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100],
                          labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    # 6. Fare binning
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

    # 7. Cabin deck extraction
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')

    return df
```

### Models to Implement

| Type | Models |
|------|--------|
| **Basic** | Logistic Regression, Decision Tree |
| **Ensemble** | Random Forest, Gradient Boosting |
| **Advanced** | XGBoost, LightGBM |
| **Hybrid** | Voting Classifier combining top 3 models |

### Model Comparison Code

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hybrid: Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, verbose=-1))
    ],
    voting='soft'
)
```

### Skills Gained
- Data cleaning and missing value handling
- Basic feature creation
- Model comparison
- Cross-validation fundamentals

---

## Project 2: House Price Prediction (Beginner-Intermediate)

### Real-World Problem
Predict residential home prices based on 79 explanatory variables — essential for real estate, banking, and insurance industries.

### Dataset
- **Source:** [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Size:** 1,460 training samples, 79 features
- **Target:** SalePrice (continuous)

### Feature Engineering Focus

```python
import numpy as np
import pandas as pd
from scipy.stats import skew

def engineer_house_features(df):
    # 1. Handle missing values (19 columns have missing data)
    # Numerical: fill with median
    # Categorical: fill with 'None' or mode

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna('None', inplace=True)

    # 2. Create TotalSF (Total Square Feet)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # 3. Create TotalBathrooms
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

    # 4. Create TotalPorchSF
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                          df['3SsnPorch'] + df['ScreenPorch'])

    # 5. House Age
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # 6. Has features (binary)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    # 7. Log-transform skewed numerical features
    numeric_feats = df.select_dtypes(include=[np.number]).columns
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75]

    for col in high_skew.index:
        df[col] = np.log1p(df[col])

    return df
```

### Models to Implement

| Type | Models |
|------|--------|
| **Linear** | Ridge, Lasso, ElasticNet |
| **Tree-Based** | Random Forest, XGBoost, LightGBM, CatBoost |
| **Hybrid** | Stacking Regressor (blend Ridge + XGBoost + LightGBM) |

### Stacking Ensemble Code

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define base models
base_models = [
    ('ridge', Ridge(alpha=10)),
    ('lasso', Lasso(alpha=0.0005)),
    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)),
    ('lgb', LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31)),
]

# Meta-learner
meta_learner = Ridge(alpha=1.0)

# Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

# Fit and predict
stacking_regressor.fit(X_train, np.log1p(y_train))
predictions = np.expm1(stacking_regressor.predict(X_test))
```

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) - Primary metric
- **RMSLE** (Root Mean Squared Log Error) - Used by Kaggle
- **R² Score** - Explained variance

### Skills Gained
- Handling high-dimensional data (79+ features)
- Feature transformation (log, polynomial)
- Regularization techniques (Ridge, Lasso)
- Stacking ensembles

---

## Project 3: Customer Churn Prediction (Intermediate)

### Real-World Problem
Predict which customers will leave a telecom/bank service — critical for customer retention strategies and reducing acquisition costs.

### Dataset
- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or [Bank Customer Churn](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)
- **Size:** ~7,000 samples
- **Class Distribution:** ~27% churn (imbalanced)

### Feature Engineering Focus

```python
def engineer_churn_features(df):
    # 1. Customer tenure segments
    df['TenureGroup'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 48, 72, np.inf],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])

    # 2. Monthly to Total Charges ratio
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # 3. Service bundle features
    services = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['ServiceCount'] = df[services].apply(
        lambda row: sum(1 for val in row if val not in ['No', 'No internet service', 'No phone service']),
        axis=1
    )

    # 4. Contract value indicator
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['ContractValue'] = df['Contract'].map(contract_map)

    # 5. Payment security (auto-pay vs manual)
    df['AutoPay'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in x.lower() else 0
    )

    # 6. High-risk customer flag
    df['HighRisk'] = ((df['Contract'] == 'Month-to-month') &
                       (df['tenure'] < 12) &
                       (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)

    return df
```

### Models to Implement

| Type | Models |
|------|--------|
| **Traditional** | Logistic Regression, SVM, KNN |
| **Ensemble** | Random Forest, Gradient Boosting, XGBoost |
| **Neural** | Simple Feed-Forward Neural Network |
| **Hybrid** | Weighted Voting Ensemble + SMOTE |

### Handling Class Imbalance

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

# SMOTE Pipeline
imb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
])

# Alternative: Class weights
from xgboost import XGBClassifier

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=4
)
```

### Evaluation Focus
- **F1-Score** - Balance precision and recall
- **ROC-AUC** - Overall discrimination ability
- **Precision-Recall Curve** - For imbalanced data
- **Threshold Optimization** - Find optimal decision boundary

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Find optimal threshold
y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Best F1-Score: {f1_scores[optimal_idx]:.4f}")
```

### Skills Gained
- Handling class imbalance (SMOTE, class weights)
- Threshold optimization
- Business metric alignment
- Customer segmentation features

---

## Project 4: Credit Card Fraud Detection (Intermediate)

### Real-World Problem
Detect fraudulent transactions from millions of legitimate ones — highly imbalanced classification with real financial impact.

### Dataset
- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Cases:** 492 (0.17%) — **Extreme imbalance!**
- **Features:** V1-V28 (PCA-transformed), Time, Amount

### Feature Engineering Focus

```python
def engineer_fraud_features(df):
    # 1. Time-based features (Time is seconds from first transaction)
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # 2. Amount transformations
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # 3. Amount bins
    df['Amount_bin'] = pd.cut(df['Amount'],
                               bins=[0, 10, 50, 100, 500, np.inf],
                               labels=['tiny', 'small', 'medium', 'large', 'huge'])

    # 4. PCA feature interactions (top components)
    df['V1_V2'] = df['V1'] * df['V2']
    df['V1_V3'] = df['V1'] * df['V3']
    df['V_sum'] = df[['V1', 'V2', 'V3', 'V4', 'V5']].sum(axis=1)

    # 5. Anomaly score using Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.001, random_state=42)
    df['IsoForest_score'] = iso_forest.fit_predict(df[['V1', 'V2', 'V3', 'V4', 'Amount']])

    return df
```

### Models to Implement

| Type | Models |
|------|--------|
| **Anomaly Detection** | Isolation Forest, One-Class SVM, Local Outlier Factor |
| **Classification** | Logistic Regression, XGBoost, LightGBM |
| **Deep Learning** | Autoencoder for anomaly detection |
| **Hybrid** | Ensemble of Isolation Forest + XGBoost + Autoencoder |

### Autoencoder for Anomaly Detection

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(input_dim, encoding_dim=14):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# Train on normal transactions only
normal_data = X_train[y_train == 0]
autoencoder = build_autoencoder(X_train.shape[1])
autoencoder.fit(normal_data, normal_data,
                epochs=50, batch_size=256,
                validation_split=0.1, verbose=0)

# Reconstruction error as anomaly score
reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.abs(X_test - reconstructions), axis=1)

# Higher error = more likely to be fraud
```

### Key Challenge: Extreme Imbalance (99.83% vs 0.17%)

```python
# Technique 1: SMOTE variants
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# Technique 2: Undersampling
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Technique 3: Combination
from imblearn.combine import SMOTETomek

# Technique 4: Cost-sensitive learning
from xgboost import XGBClassifier

# Calculate weight for minority class
weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~577

xgb_model = XGBClassifier(
    scale_pos_weight=weight,
    n_estimators=200,
    learning_rate=0.01,
    max_depth=4
)
```

### Evaluation: Use AUPRC, NOT Accuracy!

```python
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# AUPRC (Area Under Precision-Recall Curve)
y_proba = model.predict_proba(X_test)[:, 1]
auprc = average_precision_score(y_test, y_proba)
print(f"AUPRC: {auprc:.4f}")

# Visualize
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUPRC = {auprc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### Skills Gained
- Extreme imbalance handling
- Anomaly detection techniques
- Cost-sensitive learning
- Autoencoder feature extraction
- AUPRC evaluation

---

## Project 5: Store Sales Time Series Forecasting (Intermediate-Advanced)

### Real-World Problem
Predict grocery store sales across multiple stores and product families — critical for inventory and supply chain management.

### Dataset
- **Source:** [Kaggle Store Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Size:** 3+ million rows
- **Features:** Multiple stores, product families, holidays, oil prices, promotions

### Feature Engineering Focus

```python
def engineer_timeseries_features(df):
    # 1. Date features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 2. Lag features (group by store and family)
    for lag in [1, 7, 14, 28]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)

    # 3. Rolling statistics
    for window in [7, 14, 28]:
        df[f'sales_rolling_mean_{window}'] = (
            df.groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f'sales_rolling_std_{window}'] = (
            df.groupby(['store_nbr', 'family'])['sales']
            .transform(lambda x: x.shift(1).rolling(window).std())
        )

    # 4. Expanding mean (historical average)
    df['sales_expanding_mean'] = (
        df.groupby(['store_nbr', 'family'])['sales']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 5. Fourier features for seasonality
    for period in [7, 30, 365]:
        for order in range(1, 3):
            df[f'sin_{period}_{order}'] = np.sin(2 * np.pi * order * df['dayofweek'] / period)
            df[f'cos_{period}_{order}'] = np.cos(2 * np.pi * order * df['dayofweek'] / period)

    # 6. Promotional effect
    df['promo_lag_1'] = df.groupby(['store_nbr', 'family'])['onpromotion'].shift(1)
    df['promo_rolling_mean_7'] = (
        df.groupby(['store_nbr', 'family'])['onpromotion']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    return df
```

### Models to Implement

| Type | Models |
|------|--------|
| **Statistical** | ARIMA, SARIMA, Prophet |
| **ML-Based** | XGBoost, LightGBM, CatBoost |
| **Deep Learning** | LSTM, N-BEATS |
| **Hybrid** | Multi-output XGBoost + Prophet residual correction |

### Prophet + ML Hybrid

```python
from prophet import Prophet
import lightgbm as lgb

# Step 1: Fit Prophet for trend and seasonality
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
prophet_model.fit(train_df[['ds', 'y']])

# Get Prophet predictions and residuals
prophet_pred = prophet_model.predict(train_df)
residuals = train_df['y'] - prophet_pred['yhat']

# Step 2: Train LightGBM on residuals
lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01)
lgb_model.fit(X_train, residuals)

# Step 3: Combine predictions
final_pred = prophet_pred['yhat'] + lgb_model.predict(X_test)
```

### Advanced Techniques

```python
# Direct Multi-Step Forecasting
# Train separate models for each horizon
from sklearn.multioutput import MultiOutputRegressor

horizons = [1, 7, 14, 28]  # Days ahead to predict
multi_output = MultiOutputRegressor(
    lgb.LGBMRegressor(n_estimators=200)
)
multi_output.fit(X_train, y_train_multi)  # y_train_multi has 4 columns

# Hierarchical Forecasting (sum constraints)
# Store sales = Sum of all family sales
```

### Skills Gained
- Time series decomposition
- Lag and rolling features
- Handling multiple time series
- Forecasting strategies (direct vs recursive)
- Hybrid statistical + ML models

---

## Project 6: Sentiment Analysis on Reviews (Intermediate-Advanced)

### Real-World Problem
Classify customer sentiment from product/movie reviews — essential for brand monitoring, customer feedback analysis, and market research.

### Dataset
- **Source:** [IMDB 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or [Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Size:** 50,000+ reviews
- **Classes:** Positive / Negative

### Feature Engineering Focus

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download required NLTK data
nltk.download(['stopwords', 'wordnet', 'punkt'])

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenize
    tokens = text.split()

    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # 6. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

def extract_text_features(df):
    # Apply preprocessing
    df['clean_text'] = df['review'].apply(preprocess_text)

    # Text statistics
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))
    df['char_count'] = df['review'].apply(len)
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    df['sentence_count'] = df['review'].apply(lambda x: len(x.split('.')))

    # Sentiment lexicon features (using TextBlob)
    from textblob import TextBlob
    df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    return df

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=5,
    max_df=0.95
)
X_tfidf = tfidf.fit_transform(df['clean_text'])
```

### Models to Implement

| Type | Models |
|------|--------|
| **Traditional ML** | Naive Bayes, Logistic Regression, SVM |
| **Ensemble** | Random Forest, XGBoost on TF-IDF features |
| **Deep Learning** | LSTM, Bi-LSTM, CNN for text |
| **Transformers** | Fine-tuned BERT/DistilBERT |
| **Hybrid** | Ensemble of TF-IDF+XGBoost + BERT embeddings |

### LSTM for Sentiment

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

X_train_seq = tokenizer.texts_to_sequences(train_texts)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

# Build LSTM Model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)
```

### BERT Fine-tuning

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Load pre-trained DistilBERT
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors='tf'
)

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).batch(16)

# Fine-tune
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=3)
```

### Skills Gained
- NLP preprocessing pipelines
- Text vectorization (BoW, TF-IDF)
- Word embeddings (Word2Vec, GloVe)
- Transformer fine-tuning
- Multi-modal feature fusion

---

## Project 7: Chest X-Ray Pneumonia Detection (Advanced)

### Real-World Problem
Detect pneumonia from chest X-ray images — AI-assisted medical diagnosis with life-saving potential.

### Dataset
- **Source:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) or [RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- **Size:** 5,856 images
- **Classes:** Normal / Pneumonia (4:1 imbalance)

### Feature Engineering Focus

```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Image Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# 2. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 3. Traditional Feature Extraction (for ML models)
from skimage.feature import local_binary_pattern, hog

def extract_traditional_features(image):
    # Histogram features
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # LBP (Local Binary Pattern)
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, density=True)

    # HOG (Histogram of Oriented Gradients)
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate([hist, lbp_hist, hog_features[:100]])
```

### Models to Implement

| Type | Models |
|------|--------|
| **Traditional ML** | SVM/XGBoost on extracted features |
| **CNN from Scratch** | Custom CNN architecture |
| **Transfer Learning** | VGG-16, ResNet-50, DenseNet-121, EfficientNet |
| **Attention** | Vision Transformer (ViT) |
| **Hybrid** | Ensemble of ResNet + DenseNet + EfficientNet |

### Transfer Learning with ResNet50

```python
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_transfer_model(base_model_name='resnet50', num_classes=2):
    # Load pre-trained model
    if base_model_name == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'densenet121':
        base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'efficientnetb0':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base layers
    for layer in base.layers:
        layer.trainable = False

    # Add custom head
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)

    return model

# Build and compile
model = build_transfer_model('resnet50')
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune: Unfreeze last few layers after initial training
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Grad-CAM for Explainability

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_gradcam_heatmap(model, img_array, pred_index=None):
    # Get the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break

    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Visualize
def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original X-Ray')
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title('Grad-CAM Heatmap')
    plt.show()
```

### Ensemble of CNNs

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Average

# Build three models
resnet = build_transfer_model('resnet50')
densenet = build_transfer_model('densenet121')
efficientnet = build_transfer_model('efficientnetb0')

# Train each model separately...

# Ensemble: Average predictions
ensemble_output = Average()([resnet.output, densenet.output, efficientnet.output])
ensemble_model = Model(
    inputs=[resnet.input, densenet.input, efficientnet.input],
    outputs=ensemble_output
)

# Or use weighted averaging during inference
def ensemble_predict(models, weights, X):
    predictions = np.zeros((X.shape[0], 2))
    for model, weight in zip(models, weights):
        predictions += weight * model.predict(X)
    return predictions
```

### Skills Gained
- Image preprocessing and augmentation
- CNN architectures (VGG, ResNet, DenseNet, EfficientNet)
- Transfer learning mastery
- Model explainability (Grad-CAM)
- Medical AI ethics considerations
- Ensemble of deep learning models

---

## Gradient Boosting Comparison: XGBoost vs LightGBM vs CatBoost

### Quick Comparison Table

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Training Speed** | Moderate | Fast | Moderate |
| **Memory Usage** | High | Low | High |
| **Categorical Handling** | Manual encoding | Native (basic) | Native (best) |
| **Default Performance** | Good | Needs tuning | Excellent |
| **Hyperparameter Sensitivity** | High | High | Low |
| **GPU Support** | Yes | Yes | Yes |

### When to Use Each

```python
# CatBoost: When you have lots of categorical features
from catboost import CatBoostClassifier

cat_features = ['gender', 'country', 'product_type']  # Column names or indices
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=100
)

# LightGBM: When you need speed and have large datasets
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)

# XGBoost: General purpose, well-documented
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1
)
```

### Hyperparameter Tuning with Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")
```

---

## Portfolio Building Tips

### 1. Project Structure

```
project-name/
├── README.md              # Project overview, results, how to run
├── notebooks/
│   ├── 01_EDA.ipynb      # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering code
│   ├── models/           # Model training and prediction
│   └── utils/            # Helper functions
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned, transformed data
├── models/               # Saved model files
├── reports/
│   └── figures/          # Generated graphics
├── requirements.txt      # Dependencies
└── setup.py             # Package installation
```

### 2. README Template

```markdown
# Project Title

## Overview
Brief description of the problem and your solution.

## Key Results
- Model achieved XX% accuracy / 0.XX AUC
- Feature engineering improved baseline by XX%
- Best model: XGBoost with stacking ensemble

## Dataset
- Source: [Link]
- Size: X samples, Y features
- Target: Classification/Regression

## Approach
1. EDA and data cleaning
2. Feature engineering (describe key features)
3. Model comparison
4. Hyperparameter tuning
5. Ensemble creation

## Models Compared
| Model | CV Score | Test Score |
|-------|----------|------------|
| Baseline | 0.XX | 0.XX |
| XGBoost | 0.XX | 0.XX |
| Ensemble | 0.XX | 0.XX |

## Key Learnings
- What worked
- What didn't work
- Future improvements

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
```
```

### 3. Documentation Best Practices

- **Explain WHY**, not just WHAT you did
- Include visualizations for key findings
- Show model comparison tables
- Document failed experiments too (shows learning process)
- Write clean, commented code

### 4. GitHub Profile Tips

- Pin your best 6 ML projects
- Add topics/tags to repositories
- Write comprehensive READMEs
- Include demo notebooks or Streamlit apps
- Show progression from simple to complex projects

---

## Resources & References

### Official Documentation
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

### Feature Engineering
- [Kaggle Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)
- [Feature Engineering for ML - DataCamp](https://www.datacamp.com/tutorial/feature-engineering)
- [Feature Engineering Best Practices - Elite Data Science](https://elitedatascience.com/feature-engineering-best-practices)

### Model Evaluation
- [12 Important Model Evaluation Metrics - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/)
- [Performance Metrics Complete Guide - Neptune.ai](https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide)

### Kaggle Resources
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Winning a Kaggle Competition - DataCamp](https://www.datacamp.com/courses/winning-a-kaggle-competition-in-python)
- [How Top Competitors Win in 2025](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5)

### Gradient Boosting Comparison
- [When to Choose CatBoost - Neptune.ai](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm)
- [XGBoost vs CatBoost vs LightGBM - KDnuggets](https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html)
- [Kaggle Grandmaster's Playbook - NVIDIA](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

### Deep Learning
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Final Checklist

Before moving to the next project, ensure you've:

- [ ] Completed thorough EDA with visualizations
- [ ] Implemented at least 5 feature engineering techniques
- [ ] Compared 3+ different model types
- [ ] Used proper cross-validation
- [ ] Applied appropriate evaluation metrics
- [ ] Created a hybrid/ensemble model
- [ ] Documented your approach in a clean notebook
- [ ] Uploaded to GitHub with a comprehensive README
- [ ] (Optional) Written a blog post about your learnings

---

*Good luck on your machine learning journey! Remember: the key to mastery is consistent practice and learning from each project.*
