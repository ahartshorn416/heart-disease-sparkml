"""
evaluator.py
------------
Evaluation helpers that work with any binary classification PipelineModel.

Metrics reported
────────────────
  - AUC-ROC  : primary ranking metric; robust to class imbalance
  - Accuracy  : fraction of correct predictions
  - F1 Score  : harmonic mean of precision and recall

All metrics are computed on the supplied DataFrame (can be test or val).
"""

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame
import logging

logger = logging.getLogger(__name__)


def evaluate_model(predictions: DataFrame, label_col: str = "target") -> dict:
    """
    Compute AUC-ROC, Accuracy, and F1 Score on a predictions DataFrame.

    Parameters
    ----------
    predictions : DataFrame produced by PipelineModel.transform()
    label_col   : name of the ground-truth label column

    Returns
    -------
    dict with keys: auc, accuracy, f1
    """

    # ── AUC-ROC ──────────────────────────────────────────────────────────
    # BinaryClassificationEvaluator uses the rawPredictionCol (log-odds)
    # for a more precise ROC curve than the thresholded prediction column.
    binary_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = binary_eval.evaluate(predictions)

    # ── Accuracy ──────────────────────────────────────────────────────────
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )
    accuracy = acc_eval.evaluate(predictions)

    # ── F1 Score ──────────────────────────────────────────────────────────
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    )
    f1 = f1_eval.evaluate(predictions)

    metrics = {"auc": round(auc, 4), "accuracy": round(accuracy, 4), "f1": round(f1, 4)}
    logger.info("Metrics — AUC: %.4f | Accuracy: %.4f | F1: %.4f", auc, accuracy, f1)
    return metrics
