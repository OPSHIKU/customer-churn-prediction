"""
Customer Churn Prediction - End-to-End ML Pipeline
Author: Srikrith
Dataset: IBM Telco Customer Churn (7,043 customers, 21 features)

Pipeline stages:
1. Data loading & cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Preprocessing (encoding + scaling) via ColumnTransformer
5. Model training: Logistic Regression, Random Forest, XGBoost
6. Hyperparameter tuning with GridSearchCV (5-fold stratified CV)
7. Evaluation: ROC-AUC, F1-Score, Precision, Recall, RMSE (on predicted probabilities)
8. Model comparison table + best model persistence (.joblib)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    mean_squared_error, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("../data/telco_churn.csv")
print("Raw shape:", df.shape)

# TotalCharges has 11 blank strings for customers with 0 tenure -> coerce & fill
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])  # new customers: total ~= monthly

# Drop ID column (not predictive)
df = df.drop(columns=["customerID"])

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Missing values after cleaning:\n", df.isnull().sum().sum())
print("Churn rate: {:.2%}".format(df["Churn"].mean()))

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# Tenure buckets (business-interpretable segments)
df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"], include_lowest=True
)

# Average monthly spend vs total (spend trend proxy)
df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

# Count of subscribed add-on services (engagement score)
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

# Flag: month-to-month contract + electronic check (known high-risk combo)
df["high_risk_combo"] = (
    (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")
).astype(int)

# Flag: no internet-based add-ons at all (low engagement)
df["no_addons"] = (df["num_services"] == 0).astype(int)

print("Engineered features added: tenure_group, avg_charge_per_month, num_services, high_risk_combo, no_addons")
print("Shape after feature engineering:", df.shape)

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE (encoding + scaling)
# ---------------------------------------------------------------------------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                     "avg_charge_per_month", "num_services"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_features),
])

# ---------------------------------------------------------------------------
# 5 & 6. MODELS + HYPERPARAMETER GRIDS (GridSearchCV, 5-fold stratified CV)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_configs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [6, 10, None],
            "clf__min_samples_leaf": [1, 3],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
        },
    },
}

results = []
fitted_models = {}
best_overall_score = -np.inf
best_overall_name = None
best_overall_pipeline = None

for name, cfg in model_configs.items():
    print(f"\n{'='*60}\nTuning: {name}\n{'='*60}")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", cfg["estimator"])])

    grid = GridSearchCV(
        pipe, cfg["param_grid"], scoring="roc_auc", cv=cv,
        n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    fitted_models[name] = best_pipe

    # Cross-validated ROC-AUC on training folds (for reporting stability)
    cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Test set evaluation
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_proba))  # RMSE on predicted probability vs actual label

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "CV ROC-AUC (mean±std)": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        "Test ROC-AUC": round(roc_auc, 4),
        "Test F1-Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4),
        "RMSE (proba)": round(rmse, 4),
    })

    print(f"Best Params: {grid.best_params_}")
    print(f"Test ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | RMSE: {rmse:.4f}")

    if roc_auc > best_overall_score:
        best_overall_score = roc_auc
        best_overall_name = name
        best_overall_pipeline = best_pipe

# ---------------------------------------------------------------------------
# 7. COMPARISON TABLE
# ---------------------------------------------------------------------------
comparison_df = pd.DataFrame(results).sort_values("Test ROC-AUC", ascending=False).reset_index(drop=True)
print("\n\n===== MODEL COMPARISON TABLE =====")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("../reports/model_comparison.csv", index=False)

print(f"\nBest model: {best_overall_name} (Test ROC-AUC = {best_overall_score:.4f})")

# ---------------------------------------------------------------------------
# 8. PERSIST BEST MODEL
# ---------------------------------------------------------------------------
joblib.dump(best_overall_pipeline, "../models/best_churn_model.joblib")
print("Saved best model to ../models/best_churn_model.joblib")

# Also save a detailed classification report + confusion matrix for the best model
y_pred_best = best_overall_pipeline.predict(X_test)
report_txt = classification_report(y_test, y_pred_best)
with open("../reports/classification_report_best_model.txt", "w") as f:
    f.write(f"Best model: {best_overall_name}\n\n")
    f.write(report_txt)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_overall_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("../reports/confusion_matrix_best_model.png", dpi=150)
plt.close()

# ROC curves for all models
plt.figure(figsize=(7, 6))
for name, pipe in fitted_models.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/roc_curves_comparison.png", dpi=150)
plt.close()

print("\nAll artifacts saved to ../reports and ../models")
