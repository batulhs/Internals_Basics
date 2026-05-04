import argparse
import numpy as np
import joblib
import json
import os

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--condition_severity", type=float, required=True)
parser.add_argument("--patient_age", type=float, required=True)
parser.add_argument("--is_surgical", type=float, required=True)
parser.add_argument("--audiogram_score", type=float, required=True)

args = parser.parse_args()

# ---------------------------
# Load model (use best one from Task 1)
# ---------------------------
model = joblib.load("models/rf_model.pkl")

# ---------------------------
# Prepare input
# ---------------------------
features = np.array([[
    args.condition_severity,
    args.patient_age,
    args.is_surgical,
    args.audiogram_score
]])

# ---------------------------
# Predict
# ---------------------------
prediction = model.predict(features)[0]

print(f"Predicted procedure duration: {prediction:.2f} minutes")

# ---------------------------
# Save JSON result
# ---------------------------
output = {
    "image_name": "entflow-predictor",
    "image_tag": "v1",
    "base_image": "python:3.11-slim",
    "test_input": {
        "condition_severity": args.condition_severity,
        "patient_age": args.patient_age,
        "is_surgical": args.is_surgical,
        "audiogram_score": args.audiogram_score
    },
    "prediction": float(prediction)
}

os.makedirs("results", exist_ok=True)

with open("results/step3_s3.json", "w") as f:
    json.dump(output, f, indent=4)