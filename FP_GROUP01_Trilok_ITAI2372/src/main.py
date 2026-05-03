"""CLI entry point. Two subcommands: train and predict.
See docs/usage.md for examples.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import FEATURES
from .data_loader import load_dataset
from .evaluate import evaluate_model, print_report
from .predict import predict_one
from .preprocess import preprocess
from .train import save_model, train_model, train_test_split_xy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exonet",
        description="ExoNet â classify Kepler KOIs as CONFIRMED or FALSE POSITIVE.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train ------------------------------------------------------------
    sub_train = sub.add_parser("train", help="Train the model and print metrics.")
    sub_train.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the KOI table even if a cached copy exists.",
    )

    # predict ----------------------------------------------------------
    sub_pred = sub.add_parser("predict", help="Predict on a single new KOI.")
    for feat in FEATURES:
        sub_pred.add_argument(
            f"--{feat.replace('_', '-')}",
            dest=feat,
            type=float,
            required=True,
            help=f"Value for {feat}",
        )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    df = load_dataset(force_download=args.force_download)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y)

    model = train_model(X_train, y_train)
    save_model(model)

    metrics = evaluate_model(model, X_test, y_test)
    print_report(metrics)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    features = {f: getattr(args, f) for f in FEATURES}
    try:
        result = predict_one(features)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Prediction : {result['label']}")
    print(f"Probability: {result['probability']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
