import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import mlflow

# ---------------------------
# Load Data
# ---------------------------
data = pd.read_csv("data/training_data.csv")

X = data.drop("procedure_duration_min", axis=1)
y = data["procedure_duration_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Parameter Grid
# ---------------------------
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

# ---------------------------
# Base Model
# ---------------------------
rf = RandomForestRegressor(random_state=42)

# ---------------------------
# Random Search
# ---------------------------
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,  # number of trials
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    return_train_score=False
)

# ---------------------------
# MLflow Setup
# ---------------------------
mlflow.set_experiment("entflow-procedure-duration-min")

# Parent run
with mlflow.start_run(run_name="tuning-entflow") as parent_run:

    search.fit(X_train, y_train)

    results = search.cv_results_

    total_trials = len(results["params"])

    # ---------------------------
    # Log each trial as nested run
    # ---------------------------
    for i in range(total_trials):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):

            params = results["params"][i]
            mean_mae = -results["mean_test_score"][i]  # convert back

            # Log params
            mlflow.log_params(params)

            # Log metric
            mlflow.log_metric("cv_mae", mean_mae)

    # ---------------------------
    # Best Model Evaluation
    # ---------------------------
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)

    best_params = search.best_params_
    best_cv_mae = -search.best_score_

# ---------------------------
# Save JSON Output
# ---------------------------
output = {
    "search_type": "random",
    "n_folds": 5,
    "total_trials": int(total_trials),
    "best_params": best_params,
    "best_mae": float(test_mae),
    "best_cv_mae": float(best_cv_mae),
    "parent_run_name": "tuning-entflow"
}

os.makedirs("results", exist_ok=True)

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 2 completed. Results saved to results/step2_s2.json")