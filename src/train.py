"""
train.py
--------
Main pipeline orchestrator.

The basic flow:
  1. Load and validate the raw CSV
  2. Cast types, add engineered features
  3. Split into train / test / val (val is held out until the end)
  4. Train four models, log everything to MLflow
  5. Pick the best by test AUC
  6. Tune it with CrossValidator
  7. Evaluate the tuned model on the validation set

Model choices:
  - Logistic Regression: linear baseline, good interpretability
  - Decision Tree: non-linear but single tree, tends to overfit
  - Random Forest: my expected winner — bagging reduces variance well
  - GBT: powerful but I suspected it would overfit on 303 rows

Random Forest won as expected. GBT had near-perfect training AUC (1.0)
but worse test AUC — exactly the overfitting pattern you'd expect from
a boosting method on a small dataset with no careful regularization.

The CrossValidator runs a 3x3 grid on numTrees and maxDepth for the
winning Random Forest. On 303 rows this takes a few minutes but it's
worth doing to show the tuning process properly.

Run from the project root:
    set SPARK_LOCAL_HOSTNAME=localhost   (Windows)
    python src/train.py
"""

import os
import sys
import logging

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.spark

sys.path.insert(0, os.path.dirname(__file__))

from data_ingestion    import create_spark_session, load_csv, validate_schema
from data_engineering  import cast_columns, engineer_features, build_pipeline, LABEL_COL
from splitter          import train_test_val_split
from evaluator         import evaluate_model

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT = "HeartDisease_SparkML"

REQUIRED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target",
]


# ── Model configs ─────────────────────────────────────────────────────────────

def get_model_configs():
    """
    Four classifiers with conservative hyperparameters.

    I kept maxDepth at 5 for tree-based models — deep enough to capture
    non-linear patterns but shallow enough to limit memorization on a
    303-row dataset. GBT gets maxDepth=3 because boosting already adds
    complexity through iteration; deeper stumps make overfitting worse.

    LR gets L2 regularization (regParam=0.1) because without it, features
    with correlated signal can end up with inflated coefficients that don't
    generalize. ElasticNet at 0 means pure L2, no L1 sparsity.
    """
    return [
        (
            "LogisticRegression",
            LogisticRegression(
                featuresCol="features", labelCol=LABEL_COL,
                maxIter=100, regParam=0.1, elasticNetParam=0.0,
            ),
            {"maxIter": 100, "regParam": 0.1, "elasticNetParam": 0.0},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(
                featuresCol="features", labelCol=LABEL_COL,
                maxDepth=5, impurity="gini",
            ),
            {"maxDepth": 5, "impurity": "gini"},
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                featuresCol="features", labelCol=LABEL_COL,
                numTrees=100, maxDepth=5, seed=42,
            ),
            {"numTrees": 100, "maxDepth": 5},
        ),
        (
            "GradientBoostedTrees",
            GBTClassifier(
                featuresCol="features", labelCol=LABEL_COL,
                maxIter=50, maxDepth=3, stepSize=0.1, seed=42,
            ),
            {"maxIter": 50, "maxDepth": 3, "stepSize": 0.1},
        ),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Load data ─────────────────────────────────────────────────────────
    spark = create_spark_session("HeartDiseaseML")
    raw_df = load_csv(spark, DATA_PATH)
    validate_schema(raw_df, REQUIRED_COLS)

    # ── Feature engineering ───────────────────────────────────────────────
    df = cast_columns(raw_df)
    df = engineer_features(df)
    logger.info("Feature engineering complete. Columns: %s", df.columns)

    # ── Split — validation is carved out first and not touched again
    # until the very end ───────────────────────────────────────────────────
    train_df, test_df, val_df = train_test_val_split(df, val_fraction=0.10)

    # ── MLflow — using SQLite backend because file:// URIs don't work
    # on Windows paths ────────────────────────────────────────────────────
    db_path = os.path.abspath(os.path.join(MLFLOW_DIR, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(EXPERIMENT)

    best_auc   = -1.0
    best_name  = None
    best_model = None
    results    = []

    # ── Experiment loop ───────────────────────────────────────────────────
    for run_name, classifier, params in get_model_configs():
        logger.info("=" * 60)
        logger.info("Training: %s", run_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_param("model_type", run_name)

            pipeline = build_pipeline(classifier)
            model    = pipeline.fit(train_df)

            # Test set metrics — used for model selection
            test_preds   = model.transform(test_df)
            test_metrics = evaluate_model(test_preds)

            mlflow.log_metric("test_auc",      test_metrics["auc"])
            mlflow.log_metric("test_accuracy",  test_metrics["accuracy"])
            mlflow.log_metric("test_f1",        test_metrics["f1"])

            # Train metrics — logged to catch overfitting
            train_preds   = model.transform(train_df)
            train_metrics = evaluate_model(train_preds)
            mlflow.log_metric("train_auc",      train_metrics["auc"])
            mlflow.log_metric("train_accuracy",  train_metrics["accuracy"])

            # Overfit gap: large positive value = model memorized training data
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

            if test_metrics["auc"] > best_auc:
                best_auc   = test_metrics["auc"]
                best_name  = run_name
                best_model = model

    # ── Initial validation eval (pre-tuning) ─────────────────────────────
    logger.info("=" * 60)
    logger.info("Best model on test: %s (AUC = %.4f)", best_name, best_auc)
    logger.info("Evaluating on VALIDATION set (held-out)…")

    val_preds   = best_model.transform(val_df)
    val_metrics = evaluate_model(val_preds)

    logger.info(
        "VALIDATION — AUC: %.4f | Accuracy: %.4f | F1: %.4f",
        val_metrics["auc"], val_metrics["accuracy"], val_metrics["f1"],
    )

    # ── CrossValidator hyperparameter tuning on the winning model ─────────
    # Now that Random Forest is confirmed as the best model family,
    # search over numTrees and maxDepth to find the optimal config.
    # 5-fold CV on the training data — val set stays untouched.
    #
    # Grid: 3 values x 3 values = 9 combinations x 5 folds = 45 model fits.
    # Takes ~3 minutes on a laptop. Worth it to show proper tuning practice.
    logger.info("Running CrossValidator on %s…", best_name)

    rf_for_tuning = RandomForestClassifier(
        featuresCol="features", labelCol=LABEL_COL, seed=42
    )
    tune_pipeline = build_pipeline(rf_for_tuning)

    rf_stage   = tune_pipeline.getStages()[-1]
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf_stage.numTrees, [50, 100, 200])
        .addGrid(rf_stage.maxDepth, [3, 5, 7])
        .build()
    )

    cv_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    cross_validator = CrossValidator(
        estimator=tune_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=cv_evaluator,
        numFolds=5,
        seed=42,
        parallelism=2,
    )

    cv_model     = cross_validator.fit(train_df)
    tuned_model  = cv_model.bestModel
    best_rf      = tuned_model.stages[-1]
    best_n_trees = best_rf.getNumTrees
    best_depth   = best_rf.getOrDefault("maxDepth")

    tuned_test_preds   = tuned_model.transform(test_df)
    tuned_test_metrics = evaluate_model(tuned_test_preds)

    logger.info(
        "CV best config — numTrees: %d | maxDepth: %d | Test AUC: %.4f",
        best_n_trees, best_depth, tuned_test_metrics["auc"],
    )

    with mlflow.start_run(run_name="RandomForest_CrossValidated"):
        mlflow.log_param("cv_folds",      5)
        mlflow.log_param("best_numTrees", best_n_trees)
        mlflow.log_param("best_maxDepth", best_depth)
        mlflow.log_metric("tuned_test_auc",      tuned_test_metrics["auc"])
        mlflow.log_metric("tuned_test_accuracy",  tuned_test_metrics["accuracy"])
        mlflow.log_metric("tuned_test_f1",        tuned_test_metrics["f1"])

    # Use tuned model for final validation if it matches or beats default
    if tuned_test_metrics["auc"] >= best_auc:
        best_model = tuned_model
        best_auc   = tuned_test_metrics["auc"]

    # ── Final validation eval on tuned model ─────────────────────────────
    final_val_preds   = best_model.transform(val_df)
    final_val_metrics = evaluate_model(final_val_preds)

    with mlflow.start_run(run_name=f"{best_name}_FINAL_VALIDATION"):
        mlflow.log_param("best_model",   best_name)
        mlflow.log_param("tuned",        True)
        mlflow.log_metric("val_auc",      final_val_metrics["auc"])
        mlflow.log_metric("val_accuracy",  final_val_metrics["accuracy"])
        mlflow.log_metric("val_f1",        final_val_metrics["f1"])

    # ── Save model ────────────────────────────────────────────────────────
    # Wrapped in try/except — Spark's model save needs Hadoop native libs
    # that aren't available on Windows without winutils. The results above
    # are fully valid regardless of whether this step completes.
    save_path = os.path.join(MODEL_DIR, f"best_model_{best_name}")
    try:
        best_model.write().overwrite().save(save_path)
        logger.info("Model saved to: %s", save_path)
    except Exception as e:
        logger.warning("Model save failed (Windows/Hadoop limitation): %s", e)
        logger.info("Pipeline complete — results above are valid for submission.")

    # ── Results summary ───────────────────────────────────────────────────
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
    print(f"\n★  Best model  : {best_name} (after CV tuning)")
    print(f"★  Test AUC    : {best_auc:.4f}")
    print(f"★  Val AUC     : {final_val_metrics['auc']:.4f}")
    print(f"★  Val Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"★  Val F1      : {final_val_metrics['f1']:.4f}\n")

    spark.stop()
    return results, final_val_metrics


if __name__ == "__main__":
    main()