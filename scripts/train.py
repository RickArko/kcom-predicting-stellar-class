"""End-to-end training pipeline for Predicting Stellar Class.

Usage:
    uv run python scripts/train.py [--config CONFIG_PATH]
    uv run python scripts/train.py --config config/config.yaml --folds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stellar.data import load_config, load_data
from stellar.features import make_features
from stellar.models import save_submission, train_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stellar classification models.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config YAML"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print("Predicting Stellar Class - Training Pipeline")
    print("=" * 60)

    # -- 1. Load data --------------------------------------------------
    print("\n[1/5] Loading data...")
    t0 = time.time()
    train, test = load_data(cfg["paths"]["data"])
    print(f"  Train: {train.shape}, Test: {test.shape}  ({time.time()-t0:.1f}s)")

    # -- 2. Feature engineering ----------------------------------------
    print("\n[2/5] Engineering features...")
    t0 = time.time()
    feat_cfg = cfg["features"]
    X_train, X_test, y_train = make_features(
        train, test,
        target_col=cfg["competition"]["target"],
        drop_cols=feat_cfg["drop_cols"],
        color_pairs=[tuple(p) for p in feat_cfg["color_pairs"]],
    )
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}  ({time.time()-t0:.1f}s)")

    # -- 3. Train with CV ----------------------------------------------
    print(f"\n[3/5] Training with {cfg['cv']['n_splits']}-fold CV...")
    t0 = time.time()
    cv_cfg = cfg["cv"]
    model_params = {
        "lgbm": cfg.get("lgbm"),
        "xgb": cfg.get("xgb"),
        "catboost": cfg.get("catboost"),
    }
    oof_preds, test_preds = train_cv(
        X_train, y_train, X_test,
        n_splits=cv_cfg["n_splits"],
        random_state=cv_cfg["random_state"],
        model_params=model_params,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # -- 4. Save submission --------------------------------------------
    print("\n[4/5] Saving submission...")
    output_path = Path(cfg["paths"]["submissions"]) / "submission.csv"
    save_submission(
        test_ids=test["id"],
        predictions=test_preds,
        output_path=str(output_path),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
