"""Unit test for src/predict.py."""

from __future__ import annotations

import joblib
import pytest

from src.config import FEATURES
from src.data_loader import _generate_synthetic_dataset
from src.predict import predict_one
from src.preprocess import preprocess
from src.train import train_model


@pytest.fixture
def trained_model_path(tmp_path, monkeypatch):
    """Train a tiny model and point src.predict at it."""
    df = _generate_synthetic_dataset(n_per_class=80)
    X, y = preprocess(df)
    model = train_model(X, y)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr("src.train.MODEL_PATH", path)
    return path


def test_predict_one_returns_expected_shape(trained_model_path):
    sample = {f: 1.0 for f in FEATURES}
    result = predict_one(sample)
    assert set(result) == {"label", "probability", "raw"}
    assert result["label"] in {"CONFIRMED", "FALSE POSITIVE"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["raw"] in {0, 1}


def test_predict_one_rejects_missing_features(trained_model_path):
    sample = {f: 1.0 for f in FEATURES[:-1]}  # drop one
    with pytest.raises(ValueError):
        predict_one(sample)
