# Customer Churn Prediction — End-to-End ML Pipeline

Predicts which telecom customers are likely to cancel their subscription, so retention teams can target interventions before it happens.

## Dataset

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,043 customers, 21 raw features covering demographics, account details, subscribed services, and billing. ~26.5% churn rate (moderately imbalanced).

## Project Structure

```
churn_project/
├── data/
│   └── telco_churn.csv                  # Raw dataset
├── notebooks/
│   └── churn_prediction_pipeline.ipynb  # Full pipeline, executed with outputs
├── src/
│   └── pipeline.py                      # Same pipeline as a standalone script
├── models/
│   └── best_churn_model.joblib          # Trained best model (full sklearn Pipeline)
├── reports/
│   ├── model_comparison.csv             # Metrics table for all 3 models
│   ├── classification_report_best_model.txt
│   ├── confusion_matrix_best_model.png
│   ├── roc_curves_comparison.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Pipeline

1. **Data cleaning** — coerce `TotalCharges` to numeric, impute the 11 blank values (new customers with 0 tenure), drop the ID column, encode target.
2. **EDA** — class balance, tenure/charges vs. churn, churn rate by contract type.
3. **Feature engineering** — `tenure_group` buckets, `avg_charge_per_month`, `num_services` (engagement count), `high_risk_combo` (month-to-month + electronic check flag), `no_addons` flag.
4. **Preprocessing** — `ColumnTransformer` with `StandardScaler` on numeric features and `OneHotEncoder` on categorical features, wrapped in a `Pipeline` so there's no train/test leakage.
5. **Models compared** — Logistic Regression (linear baseline), Random Forest, XGBoost.
6. **Tuning** — `GridSearchCV` with 5-fold `StratifiedKFold`, optimizing ROC-AUC.
7. **Evaluation** — ROC-AUC, F1, Precision, Recall, Accuracy, and RMSE (computed on predicted probabilities vs. actual label — i.e. calibration quality, since RMSE isn't a native classification metric).

## Results

| Model               | CV ROC-AUC (mean±std) | Test ROC-AUC | Test F1-Score | Precision | Recall | Accuracy | RMSE (proba) |
|:--------------------|:-----------------------|-------------:|---------------:|----------:|-------:|---------:|-------------:|
| **XGBoost**          | 0.8474 ± 0.0112        | **0.8439**   | **0.5964**     | 0.6700    | 0.5374 | 0.8070   | 0.3682       |
| Logistic Regression | 0.8460 ± 0.0113        | 0.8415       | 0.5792         | 0.6644    | 0.5134 | 0.8020   | 0.3715       |
| Random Forest       | 0.8475 ± 0.0097        | 0.8412       | 0.5692         | 0.6803    | 0.4893 | 0.8034   | 0.3710       |

**Best model: XGBoost** (`n_estimators=200, max_depth=3, learning_rate=0.05`), selected by highest test ROC-AUC.

Full grid search results, confusion matrix, ROC curves, and feature importances are in `reports/` and in the notebook.

## Key Business Insights

- **Contract type** and **tenure** are the strongest churn predictors — month-to-month customers churn far more than those on 1–2 year contracts.
- The engineered `high_risk_combo` flag (month-to-month + electronic check) captures a segment with disproportionately high churn.
- Customers with more subscribed add-on services (`num_services`) tend to stay longer — bundling could be a retention lever.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction_pipeline.ipynb
# or run the script version:
cd src && python pipeline.py
```

## Load the Trained Model

```python
import joblib
model = joblib.load("models/best_churn_model.joblib")
predictions = model.predict(new_customer_df)          # 0 = stay, 1 = churn
probabilities = model.predict_proba(new_customer_df)[:, 1]  # churn risk score
```

The saved object is the **entire pipeline** (preprocessing + classifier), so raw customer data can be passed directly — no manual encoding/scaling needed.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib/seaborn · joblib
