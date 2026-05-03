# Usage

## Setting up

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

I built this on Python 3.10 but anything 3.10+ should work.

## Training

```
python -m src.main train
```

It will try to download the KOI cumulative table from NASA. The first run
takes maybe 10-30 seconds depending on your connection. After that the CSV
is cached in `data/` and reruns are fast.

What you should see at the end:

```
=== ExoNet evaluation ===
  test rows : 1518
  accuracy  : 0.926
  precision : 0.883
  recall    : 0.918
  F1        : 0.900
  ROC-AUC   : 0.976
  confusion :
              [901, 67]
              [45, 505]

  Top features:
    koi_prad          0.231
    koi_model_snr     0.149
    ...
```

The numbers might shift slightly if the archive updates between runs. In a
synthetic-fallback run, accuracy is closer to 0.85 â the loader will print a
warning at the top so you can tell which mode you're in.

Files written to `data/`:

- `kepler_koi.csv` â the cached NASA download
- `model.joblib` â the trained classifier
- `metrics.json` â the same numbers above as JSON
- `confusion_matrix.png` â the confusion matrix figure I use in the slides

If you want to re-download the data instead of using the cache:

```
python -m src.main train --force-download
```

## Predicting

You have to pass all ten features. Example using Kepler-22b's parameters:

```
python -m src.main predict \
    --koi-period 9.488 --koi-duration 2.95 --koi-depth 615.8 \
    --koi-prad 2.26 --koi-teq 793 --koi-insol 93.59 \
    --koi-model-snr 35.8 --koi-steff 5455 --koi-slogg 4.467 --koi-srad 0.927
```

Output:

```
Prediction : CONFIRMED
Probability: 0.965
```

If you forget to train first you'll get a clear error pointing at the
missing model file.

## Tests

```
pytest -v
```

Should be 9 tests, all passing in well under a minute.

## Common errors

If you get `ModuleNotFoundError: src`, you're probably running from inside
`src/`. `cd` back up to the repo root.

If you get an SSL error during download, you might be on a corporate proxy.
Either fix the proxy and re-run, or just accept the synthetic fallback for
this run.
