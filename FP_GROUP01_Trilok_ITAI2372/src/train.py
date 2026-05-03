"""Random Forest training. See docs/architecture.md for why I picked this
model over alternatives."""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import MODEL_PATH, RANDOM_STATE, RF_PARAMS, TEST_SIZE

log = logging.getLogger(__name__)


def train_test_split_xy(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split so both classes appear in train and test."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Fit a Random Forest with the hyperparameters from config."""
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    log.info("Trained RandomForestClassifier on %d rows", len(X_train))
    return model


def save_model(model: RandomForestClassifier, path=MODEL_PATH) -> None:
    """Persist the model so we can reload it for prediction later."""
    joblib.dump(model, path)
    log.info("Saved model to %s", path)


def load_model(path=MODEL_PATH) -> RandomForestClassifier:
    """Load a previously trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.main train` first."
        )
    return joblib.load(path)
