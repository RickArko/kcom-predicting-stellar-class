"""Inference and submission generation for Predicting Stellar Class.

Usage:
    uv run python scripts/predict.py [--config CONFIG_PATH] [--model-path MODEL_PATH]

Runs the full feature pipeline on test data and generates a submission CSV.
Currently uses the same stacked ensemble from train_cv; for production use
load a saved model ensemble from disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stellar.data import load_config, load_data
from stellar.features import make_features
from stellar.models import save_submission, train_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stellar class predictions.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model (future)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print("Predicting Stellar Class - Inference Pipeline")
    print("=" * 60)

    # -- 1. Load data --------------------------------------------------
    print("\n[1/4] Loading data...")
    train, test = load_data(cfg["paths"]["data"])

    # -- 2. Feature engineering ----------------------------------------
    print("\n[2/4] Engineering features...")
    feat_cfg = cfg["features"]
    X_train, X_test, y_train = make_features(
        train, test,
        target_col=cfg["competition"]["target"],
        drop_cols=feat_cfg["drop_cols"],
        color_pairs=[tuple(p) for p in feat_cfg["color_pairs"]],
    )

    # -- 3. Predict with ensemble --------------------------------------
    print("\n[3/4] Running ensemble prediction...")
    cv_cfg = cfg["cv"]
    model_params = {
        "lgbm": cfg.get("lgbm"),
        "xgb": cfg.get("xgb"),
        "catboost": cfg.get("catboost"),
    }
    _, test_preds = train_cv(
        X_train, y_train, X_test,
        n_splits=cv_cfg["n_splits"],
        random_state=cv_cfg["random_state"],
        model_params=model_params,
    )

    # -- 4. Save submission --------------------------------------------
    print("\n[4/4] Saving submission...")
    output_path = Path(cfg["paths"]["submissions"]) / "submission.csv"
    save_submission(
        test_ids=test["id"],
        predictions=test_preds,
        output_path=str(output_path),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
