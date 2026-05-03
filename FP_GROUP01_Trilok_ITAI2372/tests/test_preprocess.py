"""Unit tests for src/preprocess.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, LABEL_COLUMN
from src.preprocess import preprocess


def _toy_df(n: int = 8) -> pd.DataFrame:
    """Build a minimal but realistic KOI-like DataFrame."""
    rng = np.random.default_rng(0)
    data = {f: rng.normal(size=n) for f in FEATURES}
    data[LABEL_COLUMN] = ["CONFIRMED"] * (n // 2) + ["FALSE POSITIVE"] * (n - n // 2)
    return pd.DataFrame(data)


def test_preprocess_returns_xy_with_correct_shapes():
    df = _toy_df(n=10)
    X, y = preprocess(df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 10
    assert set(y.unique()).issubset({0, 1})


def test_preprocess_imputes_missing_values():
    df = _toy_df(n=6)
    df.loc[0, "koi_period"] = np.nan
    df.loc[1, "koi_depth"] = np.nan
    X, _ = preprocess(df)
    assert not X.isna().any().any(), "Imputation should have filled NaNs"


def test_preprocess_drops_unknown_labels():
    df = _toy_df(n=4)
    extra = df.iloc[:2].copy()
    extra[LABEL_COLUMN] = "CANDIDATE"  # not in LABEL_MAP
    combined = pd.concat([df, extra], ignore_index=True)
    X, y = preprocess(combined)
    # only the four CONFIRMED/FALSE POSITIVE rows should survive
    assert len(X) == 4


def test_preprocess_raises_on_empty_df():
    with pytest.raises(ValueError):
        preprocess(pd.DataFrame())


def test_preprocess_raises_on_missing_columns():
    df = _toy_df(n=4).drop(columns=["koi_period"])
    with pytest.raises(ValueError):
        preprocess(df)
