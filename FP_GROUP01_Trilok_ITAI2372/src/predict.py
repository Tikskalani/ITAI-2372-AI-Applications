"""Predict the label for a single new KOI from feature values."""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from .config import FEATURES
from .train import load_model

log = logging.getLogger(__name__)


def predict_one(features: Mapping[str, float]) -> dict:
    """Classify a single KOI given its ten input features.

    Parameters
    ----------
    features
        Dict with one entry per name in ``config.FEATURES``.

    Returns
    -------
    dict with ``label`` (str), ``probability`` (float), ``raw`` (int 0/1).
    """
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    model = load_model()
    row = pd.DataFrame([{f: float(features[f]) for f in FEATURES}])
    proba = float(model.predict_proba(row)[0, 1])
    label_int = int(proba >= 0.5)

    return {
        "label": "CONFIRMED" if label_int == 1 else "FALSE POSITIVE",
        "probability": proba,
        "raw": label_int,
    }
