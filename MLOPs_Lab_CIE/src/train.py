import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import joblib

# ---------------------------
# Load Data
# ---------------------------
data = pd.read_csv("data/training_data.csv")

X = data.drop("procedure_duration_min", axis=1)
y = data["procedure_duration_min"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MLflow Setup
# ---------------------------
mlflow.set_experiment("entflow-procedure-duration-min")

results = []

# ---------------------------
# Train SVR
# ---------------------------
with mlflow.start_run(run_name="SVR"):
    svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    
    svr.fit(X_train, y_train)
    preds = svr.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Log params
    mlflow.log_param("model", "SVR")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("epsilon", 0.1)
    mlflow.log_param("kernel", "rbf")

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Tag
    mlflow.set_tag("domain", "ent_clinic")

    # Save model
    joblib.dump(svr, "models/svr_model.pkl")

    results.append({
        "name": "SVR",
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })


# ---------------------------
# Train Random Forest
# ---------------------------
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Log params
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Tag
    mlflow.set_tag("domain", "ent_clinic")

    # Save model
    joblib.dump(rf, "models/rf_model.pkl")

    results.append({
        "name": "RandomForest",
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    })


# ---------------------------
# Select Best Model (by MAE)
# ---------------------------
best_model = min(results, key=lambda x: x["mae"])

# ---------------------------
# Save JSON Output
# ---------------------------
output = {
    "experiment_name": "entflow-procedure-duration-min",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "mae",
    "best_metric_value": best_model["mae"]
}

os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 completed. Results saved to results/step1_s1.json")