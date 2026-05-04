# MLOps Lab CIE - ENTFlow Procedure Duration Predictor

- Course: MLOps (24AM6AEMLO)
- College: BMS College of Engineering
- Semester: VI - 2026 Even
- Name: Batul H Suratwala
- USN: 1BM23AI041

---

# Project Overview

This project predicts **procedure duration (in minutes)** for ENTFlow, an ENT clinic chain.
The goal is to help optimize **operating room scheduling** by estimating how long procedures will take based on patient and clinical features.

---

# Dataset

| Feature                | Description                            |
| ---------------------- | -------------------------------------- |
| condition_severity     | Severity level (1-5)                   |
| patient_age            | Age of patient (5-80)                  |
| is_surgical            | Whether procedure is surgical (0 or 1) |
| audiogram_score        | Hearing test score (10-100)            |
| procedure_duration_min | Target - procedure duration (minutes)  |

---

# Tasks

## 🔹 Task 1 - Experiment Tracking & Model Comparison

* Trained **SVR** and **RandomForest** models
* Logged:

  * MAE
  * RMSE
  * R²
* Used MLflow experiment: `entflow-procedure-duration-min`
* Added tag: `domain = ent_clinic`
* Best model selected based on **MAE**

---

## 🔹 Task 2 - Hyperparameter Tuning

* Performed **Randomized Search** on RandomForest
* Parameter grid:

  * n_estimators: [100, 200, 300]
  * max_depth: [3, 7, 15]
  * min_samples_split: [2, 4]
* Used **5-fold cross-validation**
* Logged each trial as **nested MLflow runs** under:

  * `tuning-entflow`
* Best configuration selected using **MAE**

---

## 🔹 Task 3 - Docker Packaging

* Built CLI-based prediction tool using `argparse`
* Containerized using Docker
* Base image: `python:3.11-slim`
* Image name: `entflow-predictor:v1`
* Supports command-line prediction input
* Outputs prediction and saves JSON result

---

## 🔹 Task 4 - Retraining Pipeline

* Combined:

  * `training_data.csv` (25 rows)
  * `new_data.csv` (20 rows)
* Retrained **same model type (RandomForest)**
* Evaluated both:

  * Champion model
  * Retrained model
    (on same test set)
* Promotion rule:

  * Promote if RMSE improves by **≥ 1.0**
* Action:

  * `"promoted"` or `"kept_champion"`

---

# Results

| Task   | Output File           |
| ------ | --------------------- |
| Task 1 | results/step1_s1.json |
| Task 2 | results/step2_s2.json |
| Task 3 | results/step3_s3.json |
| Task 4 | results/step4_s8.json |

---

# How to Run

```bash
cd MLOPs_Lab_CIE

# Task 1
python src/train.py

# Task 2
python src/tune.py

# Task 3 (Docker)
docker build -t entflow-predictor:v1 .
docker run entflow-predictor:v1 --condition_severity 2 --patient_age 36 --is_surgical 0 --audiogram_score 42.6

# Task 4
python src/retrain.py
```

---

# Tech Stack

* Python 3.x
* scikit-learn
* MLflow
* pandas
* numpy
* Docker

---

# Key Features

* End-to-end MLOps pipeline
* Experiment tracking with MLflow
* Hyperparameter tuning with nested runs
* Dockerized CLI prediction tool
* Automated retraining with promotion logic

---
