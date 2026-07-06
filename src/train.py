"""
train.py
--------
Orchestrates the full ML pipeline:

  1. Load data
  2. Engineer features
  3. Split into train / test / validation
  4. Run experiments on 3 model families under MLflow tracking:
       a. Logistic Regression  (strong linear baseline)
       b. Decision Tree        (interpretable non-linear)
       c. Random Forest        (ensemble — typically best accuracy)
       d. Gradient Boosted Trees (GBT — often beats RF, slower to train)
  5. Select the best model by test AUC
  6. Evaluate the winner on the held-out validation set
  7. Save the winning PipelineModel to disk

Run from the project root:
    python src/train.py
"""

import os
import sys
import logging

# Allow MLflow to use the local file-based tracking store
# (required for MLflow >= 2.x when not using a database backend)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.spark

# ── project-level imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_ingestion    import create_spark_session, load_csv, validate_schema
from data_engineering  import cast_columns, engineer_features, build_pipeline, LABEL_COL
from splitter          import train_test_val_split
from evaluator         import evaluate_model

# ── SparkML classifiers ───────────────────────────────────────────────────────
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)

# ── Logging config ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_DIR   = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT   = "HeartDisease_SparkML"

REQUIRED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target",
]

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions — each entry is (run_name, classifier_instance, params_dict)
# params_dict is logged to MLflow so we can compare hyperparameters in the UI
# ─────────────────────────────────────────────────────────────────────────────

def get_model_configs():
    """
    Return a list of (name, classifier, params) tuples.

    We deliberately keep hyperparameters modest to reduce overfitting risk
    on this small (303-row) dataset.  A proper grid-search would be applied
    in a production setting, but here we demonstrate the multi-model
    experiment workflow cleanly.
    """
    configs = [
        (
            "LogisticRegression",
            LogisticRegression(
                featuresCol="features",
                labelCol=LABEL_COL,
                maxIter=100,
                regParam=0.1,       # L2 regularisation — prevents overfitting
                elasticNetParam=0.0,
            ),
            {"maxIter": 100, "regParam": 0.1, "elasticNetParam": 0.0},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(
                featuresCol="features",
                labelCol=LABEL_COL,
                maxDepth=5,         # shallow tree to avoid overfitting
                impurity="gini",
            ),
            {"maxDepth": 5, "impurity": "gini"},
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                featuresCol="features",
                labelCol=LABEL_COL,
                numTrees=100,
                maxDepth=5,
                seed=42,
            ),
            {"numTrees": 100, "maxDepth": 5},
        ),
        (
            "GradientBoostedTrees",
            GBTClassifier(
                featuresCol="features",
                labelCol=LABEL_COL,
                maxIter=50,
                maxDepth=3,         # shallower stumps with boosting = less overfit
                stepSize=0.1,       # learning rate
                seed=42,
            ),
            {"maxIter": 50, "maxDepth": 3, "stepSize": 0.1},
        ),
    ]
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Spark ──────────────────────────────────────────────────────────
    spark = create_spark_session("HeartDiseaseML")

    # ── 2. Load & validate ────────────────────────────────────────────────
    raw_df = load_csv(spark, DATA_PATH)
    validate_schema(raw_df, REQUIRED_COLS)

    # ── 3. Feature engineering ────────────────────────────────────────────
    df = cast_columns(raw_df)
    df = engineer_features(df)
    logger.info("Feature engineering complete. Columns: %s", df.columns)

    # ── 4. Train / test / validation split ────────────────────────────────
    # 10 % validation holdout, then 80/20 train/test on the remaining 90 %
    train_df, test_df, val_df = train_test_val_split(df, val_fraction=0.10)

    # ── 5. MLflow setup ───────────────────────────────────────────────────
    # SQLite backend works cross-platform (avoids file:// URI issues on Windows)
    db_path = os.path.abspath(os.path.join(MLFLOW_DIR, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(EXPERIMENT)

    best_auc   = -1.0
    best_name  = None
    best_model = None
    results    = []

    # ── 6. Experiment loop ────────────────────────────────────────────────
    for run_name, classifier, params in get_model_configs():
        logger.info("=" * 60)
        logger.info("Training: %s", run_name)

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters to MLflow
            mlflow.log_params(params)
            mlflow.log_param("model_type", run_name)

            # Build the full SparkML pipeline (feature eng + classifier)
            pipeline = build_pipeline(classifier)

            # Fit on training data
            model = pipeline.fit(train_df)

            # Evaluate on test set (used for model selection)
            test_preds   = model.transform(test_df)
            test_metrics = evaluate_model(test_preds)

            # Log test metrics to MLflow
            mlflow.log_metric("test_auc",      test_metrics["auc"])
            mlflow.log_metric("test_accuracy",  test_metrics["accuracy"])
            mlflow.log_metric("test_f1",        test_metrics["f1"])

            # Also evaluate on train to check for overfitting
            train_preds   = model.transform(train_df)
            train_metrics = evaluate_model(train_preds)
            mlflow.log_metric("train_auc",      train_metrics["auc"])
            mlflow.log_metric("train_accuracy",  train_metrics["accuracy"])

            # Log the overfit gap (train AUC - test AUC)
            # A large gap signals the model has memorised training data
            overfit_gap = round(train_metrics["auc"] - test_metrics["auc"], 4)
            mlflow.log_metric("overfit_gap", overfit_gap)

            logger.info(
                "%s — Test AUC: %.4f | Train AUC: %.4f | Gap: %.4f",
                run_name, test_metrics["auc"], train_metrics["auc"], overfit_gap,
            )

            results.append({
                "model":       run_name,
                "test_auc":    test_metrics["auc"],
                "train_auc":   train_metrics["auc"],
                "overfit_gap": overfit_gap,
                "accuracy":    test_metrics["accuracy"],
                "f1":          test_metrics["f1"],
            })

            # Track the best model (highest test AUC)
            if test_metrics["auc"] > best_auc:
                best_auc   = test_metrics["auc"]
                best_name  = run_name
                best_model = model

    # ── 7. Final evaluation on the validation set (held-out) ─────────────
    logger.info("=" * 60)
    logger.info("Best model on test: %s (AUC = %.4f)", best_name, best_auc)
    logger.info("Evaluating best model on VALIDATION set (held-out)…")

    val_preds   = best_model.transform(val_df)
    val_metrics = evaluate_model(val_preds)

    logger.info(
        "VALIDATION — AUC: %.4f | Accuracy: %.4f | F1: %.4f",
        val_metrics["auc"], val_metrics["accuracy"], val_metrics["f1"],
    )

    # Log validation metrics under a dedicated "final" run
    with mlflow.start_run(run_name=f"{best_name}_FINAL_VALIDATION"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("val_auc",      val_metrics["auc"])
        mlflow.log_metric("val_accuracy",  val_metrics["accuracy"])
        mlflow.log_metric("val_f1",        val_metrics["f1"])

    # ── 8. Save best model ────────────────────────────────────────────────
    # Wrapped in try/except because saving requires Hadoop native libraries
    # which are not available on all Windows setups. The pipeline results
    # above are fully valid regardless of whether the model saves to disk.
    save_path = os.path.join(MODEL_DIR, f"best_model_{best_name}")
    try:
        best_model.write().overwrite().save(save_path)
        logger.info("Best model saved to: %s", save_path)
    except Exception as e:
        logger.warning(
            "Could not save model to disk (common on Windows without winutils): %s", e
        )
        logger.info("Pipeline complete — results above are fully valid for submission.")

    # ── 9. Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Test AUC':>9} {'Train AUC':>10} {'Gap':>7} {'Accuracy':>10} {'F1':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model']:<25} {r['test_auc']:>9.4f} {r['train_auc']:>10.4f}"
            f" {r['overfit_gap']:>7.4f} {r['accuracy']:>10.4f} {r['f1']:>7.4f}"
        )
    print("=" * 70)
    print(f"\n★  Best model  : {best_name}")
    print(f"★  Test AUC    : {best_auc:.4f}")
    print(f"★  Val AUC     : {val_metrics['auc']:.4f}")
    print(f"★  Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"★  Val F1      : {val_metrics['f1']:.4f}\n")

    spark.stop()
    return results, val_metrics


if __name__ == "__main__":
    main()
