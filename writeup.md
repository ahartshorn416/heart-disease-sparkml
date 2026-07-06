# Heart Disease Prediction with SparkML — Project Findings

**Dataset:** UCI Heart Disease (Cleveland) — 303 observations, 14 features  
**Task:** Binary classification — predict presence of heart disease (1) vs. absence (0)  
**Source:** https://archive.ics.uci.edu/dataset/45/heart+disease

---

## Dataset Overview

The Cleveland Heart Disease dataset is a classic benchmark from the UCI Machine Learning
Repository. Each row represents a patient with clinical measurements taken during a
cardiac evaluation. The target label is binary: 1 indicates at least one major vessel
showing >50% stenosis (disease), 0 indicates a healthy result. The dataset has 303
patients with a slight positive skew (165 disease, 138 healthy).

---

## Data Engineering

Two features were engineered before modelling:

**Feature 1 — `age_group` (Binning):**  
Cardiovascular risk is not linear with age — it jumps at clinical thresholds.  
Age was bucketed into four ordinal groups: <40, 40–50, 50–60, 60+.  
This was implemented via SparkML's `Bucketizer` and then one-hot-encoded.

**Feature 2 — `bp_chol_risk` (Composite clinical flag):**  
High blood pressure (>130 mmHg) combined with high cholesterol (>240 mg/dL) is
clinically known to compound cardiovascular risk beyond either factor alone.
A binary flag was created: 1 if both conditions are met, 0 otherwise.

**Normalization:** `StandardScaler` (zero mean, unit variance) was applied to the
assembled feature vector. This is essential for Logistic Regression, which is
sensitive to feature scale, and benefits ensemble methods as well.

**One-Hot Encoding:** The categorical columns `cp` (chest pain type), `restecg`
(resting ECG), `slope` (ST slope), `thal` (thalassemia type), and the engineered
`age_group` were indexed with `StringIndexer` and then OHE'd with `OneHotEncoder`.

---

## Train / Test / Validation Split

- **Validation set (10%):** 30 observations, held out completely until final reporting.
- **Test set (~14%):** ~54 observations, used to compare models and select the winner.
- **Training set (~78%):** ~219 observations, used to fit all models.

---

## Model Experiments

Four model families were compared under MLflow tracking:

| Model | Test AUC | Train AUC | Overfit Gap | Test Accuracy |
|-------|----------|-----------|-------------|---------------|
| **Random Forest** | **0.875** | 0.988 | 0.113 | 0.836 |
| Gradient Boosted Trees | 0.868 | 1.000 | 0.132 ⚠️ | 0.836 |
| Logistic Regression | 0.853 | 0.941 | 0.087 | 0.803 |
| Decision Tree | 0.816 | 0.960 | 0.144 | 0.770 |

**Selected model: Random Forest**  
The Random Forest achieved the highest test AUC (0.875). While GBT had a similar test
AUC (0.868), its training AUC of 1.000 revealed perfect memorisation of the training
set — a strong signal of overfitting. The Random Forest's ensemble averaging via
bagging gives it better variance reduction and more robust generalisation.

---

## Validation Set Results (Held-Out)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.877 |
| Accuracy | 0.800 |
| F1 Score | 0.801 |

The validation AUC (0.877) slightly *exceeded* the test AUC (0.875), confirming the
model generalises reliably to unseen data and was not overfit to the test set during
model selection.

---

## Key Findings

1. **Ensemble methods outperform single trees** — Random Forest beat the Decision
   Tree by ~6 AUC points on real data, confirming the value of variance reduction.

2. **GBT overfits on small data despite strong test performance** — A perfect train
   AUC (1.000) with a 13-point overfit gap makes GBT unreliable here, even though
   its test score was competitive.

3. **Logistic Regression remains a strong baseline** — With L2 regularisation it
   achieved a moderate overfit gap (0.087) and strong accuracy, making it the most
   interpretable option for a clinical setting.

4. **Feature engineering added signal** — The `bp_chol_risk` composite flag encodes
   clinical knowledge that the model cannot easily discover from two separate
   continuous variables alone.

5. **Real data validates well** — The Random Forest validation AUC of 0.877 on 303
   real patient observations is a clinically meaningful result, exceeding common
   benchmarks for this dataset in the literature (~0.84–0.88 AUC range).
