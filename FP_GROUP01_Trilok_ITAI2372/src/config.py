"""Constants and paths used by the rest of the project."""

from __future__ import annotations

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "kepler_koi.csv"
MODEL_PATH = DATA_DIR / "model.joblib"
METRICS_PATH = DATA_DIR / "metrics.json"
CONFUSION_PNG = DATA_DIR / "confusion_matrix.png"

# NASA Exoplanet Archive, KOI cumulative table as CSV.
# Column reference:
# https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
KOI_DOWNLOAD_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    "?table=cumulative&select=*&format=csv"
)

# Features I picked. See docs/data_dictionary.md for what each one is.
FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

LABEL_COLUMN = "koi_disposition"

# Drop CANDIDATE rows. Map the rest to 0/1.
LABEL_MAP = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
}

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.20

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
