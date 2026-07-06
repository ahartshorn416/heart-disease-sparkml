"""
splitter.py
-----------
Creates reproducible train / test / validation splits.

Split strategy
──────────────
  - 10 % → validation  (held out entirely until final reporting)
  - 72 % → train       (used for model fitting)
  - 18 % → test        (used for hyperparameter selection / experiment comparison)

Total: 100 %

We split in two passes so the 10 % holdout is isolated first,
then the remaining 90 % is divided 80/20 for train/test.
This gives roughly a 72 / 18 / 10 split overall.

Seed is fixed at 42 throughout the project for reproducibility.
"""

from pyspark.sql import DataFrame
import logging

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def train_test_val_split(
    df: DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> tuple:
    """
    Split a DataFrame into train, test, and validation sets.

    Parameters
    ----------
    df            : input Spark DataFrame (pre-feature-engineering)
    val_fraction  : fraction of total data reserved for validation (default 10 %)
    test_fraction : fraction of the *remaining* data used for test  (default 20 %)

    Returns
    -------
    (train_df, test_df, val_df) — three non-overlapping DataFrames
    """

    # ── Pass 1: carve out validation set ────────────────────────────────
    train_test_fraction = 1.0 - val_fraction
    train_test_df, val_df = df.randomSplit(
        [train_test_fraction, val_fraction], seed=RANDOM_SEED
    )

    # ── Pass 2: split the remainder into train / test ───────────────────
    train_fraction = 1.0 - test_fraction
    train_df, test_df = train_test_df.randomSplit(
        [train_fraction, test_fraction], seed=RANDOM_SEED
    )

    # Log counts so we can verify the proportions are reasonable
    total   = df.count()
    n_train = train_df.count()
    n_test  = test_df.count()
    n_val   = val_df.count()

    logger.info(
        "Split complete — total: %d | train: %d (%.0f%%) | "
        "test: %d (%.0f%%) | val: %d (%.0f%%)",
        total,
        n_train, 100 * n_train / total,
        n_test,  100 * n_test  / total,
        n_val,   100 * n_val   / total,
    )

    return train_df, test_df, val_df
