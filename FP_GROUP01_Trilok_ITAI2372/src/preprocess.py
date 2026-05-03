"""Cleaning step. Turn the raw KOI rows into numeric X and y.

Steps: coerce to numeric, drop empty rows, median-impute, map the label.
Median imputation is fine because Random Forests split by quantile so
they do not really care about the imputation strategy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import FEATURES, LABEL_COLUMN, LABEL_MAP

log = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X has float features, y is 0/1 ints."""
    if df.empty:
        raise ValueError("preprocess() got an empty DataFrame.")

    missing = [c for c in FEATURES + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()

    # 1) coerce features to numeric
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) drop rows with no useful features
    before = len(df)
    df = df.dropna(subset=FEATURES, how="all")
    if len(df) < before:
        log.info("Dropped %d rows where all features were NaN", before - len(df))

    # 3) median impute per column
    for col in FEATURES:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # 4) label encoding
    df = df[df[LABEL_COLUMN].isin(LABEL_MAP)]
    y = df[LABEL_COLUMN].map(LABEL_MAP).astype(int)
    X = df[FEATURES].astype(float).reset_index(drop=True)
    y = y.reset_index(drop=True)

    if X.empty:
        raise ValueError("After preprocessing, no rows remained.")

    log.info("Preprocessed dataset: %d rows, %d features", len(X), X.shape[1])
    return X, y
