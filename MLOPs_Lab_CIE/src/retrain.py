import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------
# Load original data
# ---------------------------
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# ---------------------------
# Combine datasets
# ---------------------------
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# ---------------------------
# Features & target
# ---------------------------
X_original = train_df.drop("procedure_duration_min", axis=1)
y_original = train_df["procedure_duration_min"]

X_combined = combined_df.drop("procedure_duration_min", axis=1)
y_combined = combined_df["procedure_duration_min"]

# ---------------------------
# SAME TEST SET (IMPORTANT)
# ---------------------------
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

# ---------------------------
# Load champion model
# ---------------------------
champion_model = joblib.load("models/rf_model.pkl")

# Evaluate champion
champion_preds = champion_model.predict(X_test)
champion_rmse = np.sqrt(mean_squared_error(y_test, champion_preds))

# ---------------------------
# Retrain model on combined data
# ---------------------------
retrained_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

retrained_model.fit(X_combined, y_combined)

# Evaluate retrained model on SAME test set
retrained_preds = retrained_model.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, retrained_preds))

# ---------------------------
# Compare
# ---------------------------
improvement = champion_rmse - retrained_rmse
threshold = 1.0

if improvement >= threshold:
    action = "promoted"
    # Save new model as champion
    joblib.dump(retrained_model, "models/rf_model.pkl")
else:
    action = "kept_champion"

# ---------------------------
# Save JSON
# ---------------------------
output = {
    "original_data_rows": int(len(train_df)),
    "new_data_rows": int(len(new_df)),
    "combined_data_rows": int(len(combined_df)),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 4 completed. Results saved to results/step4_s8.json")