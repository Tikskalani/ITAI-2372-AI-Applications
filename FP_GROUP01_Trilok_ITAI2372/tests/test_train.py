"""Unit tests for src/train.py and src/evaluate.py."""

from __future__ import annotations

import json

from src.config import METRICS_PATH, MODEL_PATH
from src.data_loader import _generate_synthetic_dataset
from src.evaluate import evaluate_model
from src.preprocess import preprocess
from src.train import save_model, train_model, train_test_split_xy


def test_end_to_end_synthetic_pipeline(tmp_path, monkeypatch):
    """Train + evaluate + save round-trips on synthetic data."""
    df = _generate_synthetic_dataset(n_per_class=120)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y)

    model = train_model(X_train, y_train)

    # redirect MODEL_PATH and METRICS_PATH into tmp_path so we don't pollute the repo
    monkeypatch.setattr("src.train.MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.evaluate.METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr("src.evaluate.CONFUSION_PNG", tmp_path / "cm.png")

    save_model(model, path=tmp_path / "model.joblib")
    metrics = evaluate_model(model, X_test, y_test)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cm.png").exists()
    # synthetic data is engineered to be moderately separable
    assert metrics["accuracy"] > 0.7, metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_split_is_stratified():
    df = _generate_synthetic_dataset(n_per_class=100)
    X, y = preprocess(df)
    _, _, y_tr, y_te = train_test_split_xy(X, y)
    # ratios should be close after stratified split
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
