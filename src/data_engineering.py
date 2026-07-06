"""
data_engineering.py
--------------------
All feature engineering, normalization, and encoding steps live here.

Pipeline stages applied in order
─────────────────────────────────
1.  Cast columns to correct types (safety net after CSV ingestion)
2.  Engineer Feature 1 → age_group   (binning continuous age into ordinal buckets)
3.  Engineer Feature 2 → bp_chol_risk  (composite risk score from BP and cholesterol)
4.  One-hot-encode categorical columns: cp, restecg, slope, thal, age_group
5.  Assemble all numeric + OHE columns into a single 'features' vector
6.  Normalize the feature vector with StandardScaler (zero mean, unit variance)

The function build_pipeline() returns a configured (but unfitted) SparkML Pipeline
so that fitting and transforming can be done cleanly in train.py.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    Bucketizer,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import logging

logger = logging.getLogger(__name__)

# ── Column groups ────────────────────────────────────────────────────────────

# Columns that are categorical and need OHE
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]

# Raw numeric features (before feature engineering additions)
NUMERIC_COLS = [
    "age", "sex", "trestbps", "chol", "fbs",
    "thalach", "exang", "oldpeak", "ca",
]

LABEL_COL = "target"


# ── Step 1: type casting ─────────────────────────────────────────────────────

def cast_columns(df: DataFrame) -> DataFrame:
    """
    Ensure all feature columns are cast to DoubleType, which SparkML requires.
    The target label is cast to DoubleType as well (classifiers expect a double).
    """
    for col in NUMERIC_COLS + CATEGORICAL_COLS + [LABEL_COL]:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))
    logger.info("Column types cast to DoubleType.")
    return df


# ── Step 2 & 3: Feature Engineering ─────────────────────────────────────────

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Create two new engineered features before building the ML pipeline.

    Feature 1 — age_group (binning):
        Age is continuous but cardiovascular risk jumps at known thresholds.
        We bin age into 4 buckets: <40, 40-50, 50-60, 60+
        This converts a smooth numeric into an ordinal categorical that
        the model can treat non-linearly.

    Feature 2 — bp_chol_risk (composite score):
        High blood pressure AND high cholesterol together are a stronger
        predictor than either alone. We create a single risk indicator:
            bp_chol_risk = 1 if (trestbps > 130 AND chol > 240) else 0
        This encodes clinical domain knowledge as a binary flag.
    """

    # --- Feature 1: age group bins ---
    # Bucketizer requires Double input and produces a Double index
    # Bucket splits: (-inf, 40), [40, 50), [50, 60), [60, +inf)
    df = df.withColumn("age_dbl", F.col("age").cast(DoubleType()))

    # We'll handle this in the pipeline via Bucketizer (see build_pipeline)
    # but first add a raw numeric version as a placeholder so the schema is stable.

    # --- Feature 2: composite BP + cholesterol risk flag ---
    df = df.withColumn(
        "bp_chol_risk",
        F.when(
            (F.col("trestbps") > 130) & (F.col("chol") > 240), 1.0
        ).otherwise(0.0)
    )

    logger.info("Engineered features added: age_dbl, bp_chol_risk")
    return df


# ── Step 4 & 5: SparkML Pipeline ─────────────────────────────────────────────

def build_pipeline(classifier) -> Pipeline:
    """
    Construct the full feature-engineering + classification Pipeline.

    Parameters
    ----------
    classifier : a fitted-able SparkML estimator (e.g. LogisticRegression)

    Returns
    -------
    Pipeline (unfitted) — call .fit(train_df) to get a PipelineModel.

    Stage breakdown
    ───────────────
    Stage 0  — Bucketizer: bin age_dbl → age_group_idx (0,1,2,3)
    Stage 1  — StringIndexer × 4: index each categorical col (cp, restecg, slope, thal)
    Stage 2  — OneHotEncoder × 5: OHE the indexed categoricals + age_group_idx
    Stage 3  — VectorAssembler: gather all numeric + OHE vectors into 'raw_features'
    Stage 4  — StandardScaler: normalize raw_features → 'features'
    Stage 5  — Classifier: fit/predict on 'features'
    """

    stages = []

    # ── Stage 0: Bin age into groups ─────────────────────────────────────
    # splits define the bucket boundaries; -inf and +inf as outer bounds
    age_bucketizer = Bucketizer(
        splits=[-float("inf"), 40.0, 50.0, 60.0, float("inf")],
        inputCol="age_dbl",
        outputCol="age_group_idx",
    )
    stages.append(age_bucketizer)

    # ── Stage 1: StringIndexer for categoricals ───────────────────────────
    # SparkML's OHE requires integer indices, not raw numeric values.
    # Even though our categoricals are already integers (0,1,2…), StringIndexer
    # ensures a consistent mapping and handles unseen values gracefully.
    indexer_output_cols = [f"{c}_idx" for c in CATEGORICAL_COLS]
    string_indexers = [
        StringIndexer(inputCol=c, outputCol=out, handleInvalid="keep")
        for c, out in zip(CATEGORICAL_COLS, indexer_output_cols)
    ]
    stages.extend(string_indexers)

    # ── Stage 2: OneHotEncoder ─────────────────────────────────────────────
    # OHE all categorical indices (including the engineered age_group_idx)
    ohe_input_cols  = indexer_output_cols + ["age_group_idx"]
    ohe_output_cols = [f"{c}_vec" for c in CATEGORICAL_COLS] + ["age_group_vec"]
    ohe = OneHotEncoder(inputCols=ohe_input_cols, outputCols=ohe_output_cols)
    stages.append(ohe)

    # ── Stage 3: VectorAssembler ──────────────────────────────────────────
    # Combine raw numerics + bp_chol_risk + OHE vectors into one feature vector
    assembler_inputs = NUMERIC_COLS + ["bp_chol_risk"] + ohe_output_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    # ── Stage 4: StandardScaler ───────────────────────────────────────────
    # Normalize to zero mean, unit std. Critical for Logistic Regression and SVM;
    # doesn't hurt tree-based models either.
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    stages.append(scaler)

    # ── Stage 5: Classifier ───────────────────────────────────────────────
    stages.append(classifier)

    pipeline = Pipeline(stages=stages)
    logger.info("Pipeline built with %d stages + classifier.", len(stages))
    return pipeline
