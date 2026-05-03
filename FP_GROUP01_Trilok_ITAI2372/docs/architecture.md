# Architecture notes

This is a small project so the architecture is more about "what does each
file do" than any deep system design.

## Modules

`config.py` holds the constants. Paths, the list of feature column names,
the label mapping, and the Random Forest hyperparameters are all in here so I
don't have to hunt for them when tuning.

`data_loader.py` downloads the Kepler KOI cumulative table from the NASA
Exoplanet Archive and caches it to `data/kepler_koi.csv`. If the download
fails (no network, archive moved, SSL issue), it prints a warning and
generates a synthetic dataset so the rest of the pipeline can still run. I
wanted this fallback so the code is gradeable offline.

`preprocess.py` takes the raw DataFrame and returns `(X, y)` with float
features and int labels. The cleaning is pretty light: coerce strings to
numbers, drop rows with no usable features at all, fill remaining NaNs with
the column median, then drop CANDIDATE rows since their label is undecided.

`train.py` does an 80/20 stratified split, fits a RandomForestClassifier, and
saves the model to disk with joblib.

`evaluate.py` computes accuracy, precision, recall, F1, and ROC-AUC, dumps
them to `data/metrics.json`, and writes a confusion matrix PNG that I use in
the slides.

`predict.py` is the inference path. Given a dict of feature values, it loads
the saved model and returns a label and probability.

`main.py` is the CLI. Two subcommands: `train` and `predict`.

## Data flow

1. `data_loader.load_dataset()` returns the DataFrame.
2. `preprocess.preprocess()` returns `(X, y)`.
3. `train.train_test_split_xy()` splits, `train.train_model()` fits.
4. `train.save_model()` persists.
5. `evaluate.evaluate_model()` writes metrics and the confusion matrix.
6. At inference time `predict.predict_one()` loads the model and scores one
   row.

## Why a Random Forest

I went back and forth on this. A neural network felt like overkill for a
tabular dataset of ~7,500 rows, and the Kepler community itself (e.g. the
Robovetter pipeline) leans on tree-based methods. Random Forest doesn't need
feature scaling, handles non-linear feature interactions on its own, and
gives you feature importances which I want to talk about in the
presentation. It also trains in a few seconds so the iteration loop is fast.

Possible follow-ups if I had more time: try gradient boosting (XGBoost or
LightGBM) and compare on the same split, or add cross-validation instead of
the single hold-out test.

## A few decisions and why

- 10 features rather than the full 50+: most of the extra columns are flags
  or duplicates of the same physical quantity. I picked ten that are dense
  (most rows have them) and that map onto things astronomers actually look at.
- Median imputation: trees don't really care, so this is fine and avoids
  pulling in a ColumnTransformer.
- Random seed everywhere: makes the training reproducible for grading.
- CLI rather than a notebook or web UI: keeps the project small and easy to
  run.
