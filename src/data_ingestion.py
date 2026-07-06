"""
data_ingestion.py
-----------------
Responsible for loading the raw Heart Disease dataset into a Spark DataFrame.

Dataset: UCI Heart Disease (Cleveland)
Source:  https://archive.ics.uci.edu/dataset/45/heart+disease
Target:  binary — 1 = disease present, 0 = no disease

This module is intentionally thin; all transformation logic lives in
data_engineering.py to keep concerns separated.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
import logging

logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "HeartDiseaseML") -> SparkSession:
    """
    Create (or retrieve) a local SparkSession.

    We use 'local[*]' so Spark uses all available CPU cores.
    Log level is set to WARN to reduce console noise.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")   # small dataset → fewer partitions
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created: %s", app_name)
    return spark


def load_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Load a CSV file with a header row into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
    path  : str — absolute path to the CSV file

    Returns
    -------
    DataFrame with inferred schema (all columns still as strings at this stage).
    """
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")   # let Spark infer int/double types
        .csv(path)
    )
    logger.info("Loaded %d rows from %s", df.count(), path)
    return df


def validate_schema(df: DataFrame, required_cols: list) -> None:
    """
    Assert that every expected column is present in the DataFrame.
    Raises ValueError if any column is missing — fail fast rather than
    surfacing cryptic downstream errors.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    logger.info("Schema validation passed — all required columns present.")
