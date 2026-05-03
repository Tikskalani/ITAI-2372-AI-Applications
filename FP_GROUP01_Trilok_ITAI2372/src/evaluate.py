"""Score the trained classifier and write metrics + confusion matrix to disk."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless â no display required
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CONFUSION_PNG, FEATURES, METRICS_PATH

log = logging.getLogger(__name__)


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Score the model and write metrics + confusion matrix to disk."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": _feature_importance(model),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Wrote metrics to %s", METRICS_PATH)

    _save_confusion_plot(metrics["confusion_matrix"], CONFUSION_PNG)
    return metrics


def _feature_importance(model: RandomForestClassifier) -> list[dict]:
    pairs = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"feature": name, "importance": float(imp)} for name, imp in pairs]


def _save_confusion_plot(matrix, out_path) -> None:
    """Render a confusion matrix PNG for the slides."""
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=_as_2d(matrix),
        display_labels=["FALSE POS", "CONFIRMED"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote confusion matrix to %s", out_path)


def _as_2d(matrix):
    """Allow tests to pass plain lists or numpy arrays interchangeably."""
    import numpy as np

    return np.asarray(matrix)


def print_report(metrics: dict[str, Any]) -> None:
    """Pretty-print metrics for the CLI user."""
    print("\n=== ExoNet evaluation ===")
    print(f"  test rows : {metrics['n_test']}")
    print(f"  accuracy  : {metrics['accuracy']:.3f}")
    print(f"  precision : {metrics['precision']:.3f}")
    print(f"  recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.3f}")
    print("  confusion :")
    for row in metrics["confusion_matrix"]:
        print("              ", row)
    print("\n  Top features:")
    for entry in metrics["feature_importance"][:5]:
        print(f"    {entry['feature']:<18}{entry['importance']:.3f}")
    print()
