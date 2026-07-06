# 🫀 Heart Disease Prediction with SparkML

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.4%2B-E25A1C?logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning pipeline built with **Apache SparkML** that predicts the presence of heart disease from clinical patient data. The project covers the full ML lifecycle — data engineering, feature engineering, multi-model experimentation, and experiment tracking with MLflow.

**Best model:** Random Forest · **Test AUC: 0.875** · **Val AUC: 0.877**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [MLflow Tracking](#mlflow-tracking)

---

## Overview

This project uses the [UCI Heart Disease (Cleveland) dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) to build a **binary classifier** that predicts whether a patient has heart disease based on 14 clinical features (age, chest pain type, blood pressure, cholesterol, etc.).

Four model families are trained and compared under **MLflow experiment tracking**. The best model is selected by test AUC and evaluated on a 10% held-out validation set to give an unbiased final performance estimate.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Observations | 303 patients |
| Features | 14 (mix of continuous and categorical) |
| Target | Binary — 1 = heart disease, 0 = healthy |
| Class balance | ~54% positive, ~46% negative |

Key features include: `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate), `exang` (exercise-induced angina), `oldpeak` (ST depression), and `thal` (thalassemia type).

---

## Pipeline Architecture

```
Raw CSV
   │
   ▼
data_ingestion.py ──── Load & validate schema
   │
   ▼
data_engineering.py ── Cast types
                   ──── Engineer: age_group (Bucketizer)
                   ──── Engineer: bp_chol_risk (composite flag)
   │
   ▼
splitter.py ─────────── 78% train / 14% test / 10% validation (held out)
   │
   ▼
SparkML Pipeline ─────── StringIndexer → OneHotEncoder
                          → VectorAssembler → StandardScaler
                          → Classifier
   │
   ▼
MLflow Tracking ──────── Log params, metrics, overfit gap per run
   │
   ▼
evaluator.py ─────────── AUC-ROC · Accuracy · F1 on validation set
```

---

## Feature Engineering

Two features were engineered from the raw data before modelling:

**`age_group` — Binning (SparkML `Bucketizer`)**

Age is continuous but cardiovascular risk increases non-linearly at clinical thresholds. Age is bucketed into four ordinal groups, then one-hot-encoded:

| Bucket | Range |
|--------|-------|
| 0 | Under 40 |
| 1 | 40 – 50 |
| 2 | 50 – 60 |
| 3 | 60 and over |

**`bp_chol_risk` — Composite clinical flag**

High blood pressure and high cholesterol together compound risk beyond either factor alone. This binary flag encodes clinical domain knowledge:

```python
bp_chol_risk = 1  if  trestbps > 130  AND  chol > 240  else  0
```

**Normalization:** `StandardScaler` (zero mean, unit variance) is applied to the assembled feature vector — essential for Logistic Regression and beneficial for all models.

**One-hot encoding:** `StringIndexer` + `OneHotEncoder` applied to `cp`, `restecg`, `slope`, `thal`, and the engineered `age_group`.

---

## Model Results

Four model families were compared. The winning model is selected by **test AUC** (not training AUC) to avoid selecting an overfit model.

| Model | Test AUC | Train AUC | Overfit Gap | Accuracy | F1 |
|-------|:--------:|:---------:|:-----------:|:--------:|:--:|
| **Random Forest** ⭐ | **0.875** | 0.988 | 0.113 | 0.836 | 0.836 |
| Gradient Boosted Trees | 0.868 | 1.000 | 0.132 ⚠️ | 0.836 | 0.836 |
| Logistic Regression | 0.853 | 0.941 | 0.087 | 0.803 | 0.803 |
| Decision Tree | 0.816 | 0.960 | 0.144 | 0.770 | 0.770 |

**Final validation set (10% holdout):**

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.877 |
| Accuracy | 0.800 |
| F1 Score | 0.801 |

**Key observations:**
- Random Forest achieved the best test AUC (0.875) and validated strongly at 0.877 — no sign of overfitting to the test set
- GBT memorised the training data perfectly (AUC = 1.000) but its test AUC was slightly lower than Random Forest — classic overfitting on a small dataset
- Logistic Regression was the most interpretable model and competitive across all metrics
- The validation AUC (0.877) actually slightly *exceeded* the test AUC (0.875), confirming the model generalises well to unseen data

---

## Project Structure

```
heart-disease-sparkml/
├── data/
│   └── heart_disease.csv          # UCI Cleveland Heart Disease dataset
├── mlruns/                        # MLflow experiment tracking store
├── screenshots/
│   └── README.md                  # MLflow UI setup instructions
├── src/
│   ├── data_ingestion.py          # SparkSession creation, CSV loading, schema validation
│   ├── data_engineering.py        # Feature engineering, OHE, scaling, pipeline builder
│   ├── splitter.py                # Train / test / validation split (10% holdout)
│   ├── evaluator.py               # AUC, Accuracy, F1 evaluation helpers
│   └── train.py                   # Main orchestrator — experiments + MLflow tracking
├── .gitignore
├── requirements.txt
├── writeup.md                     # One-page project findings
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Java 8 or 11 (required by Apache Spark)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/heart-disease-sparkml.git
cd heart-disease-sparkml

# Install dependencies
pip install -r requirements.txt
```

### Run the pipeline

```bash
python src/train.py
```

This will:
1. Load and engineer features from `data/heart_disease.csv`
2. Split into train / test / validation sets
3. Train 4 models, logging all metrics to MLflow
4. Print a results summary table
5. Save the best model to `models/`

---

## MLflow Tracking

All experiments are tracked locally. To explore the interactive tracking UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open **http://localhost:5000** in your browser.

The UI shows all 5 runs (4 model experiments + final validation), with params, metrics, and overfit gap logged per run — making it easy to compare models and spot overfitting at a glance.

---

## Tech Stack

- **Apache Spark 3.4** — distributed data processing and ML pipeline
- **SparkML** — Bucketizer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, classifiers
- **MLflow 2.10** — experiment tracking, parameter logging, metric comparison
- **Python 3.9+** — modular, well-commented codebase

---

## License

MIT
