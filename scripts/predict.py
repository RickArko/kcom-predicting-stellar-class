"""Inference from a saved ensemble for Predicting Stellar Class.

Usage:
    uv run python scripts/predict.py --run-dir outputs/runs/20260624_143042_experiment-name
    uv run python scripts/predict.py --run-dir outputs/runs/latest  # symlink works too
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from stellar.data import load_config, load_data
from stellar.features import ColorFeatureEngineer
from stellar.models import StackingEnsemble, save_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submissions from a trained ensemble.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to an experiment run directory containing models/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for submission CSV (default: run-dir/submission.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "models" / "ensemble.joblib"

    if not config_path.exists():
        logger.error("Config not found at %s", config_path)
        raise SystemExit(1)
    if not model_path.exists():
        logger.error("Ensemble model not found at %s", model_path)
        raise SystemExit(1)

    cfg = load_config(str(config_path))

    # -- 1. Load data --------------------------------------------------
    logger.info("[1/4] Loading data ...")
    train, test = load_data(cfg["paths"]["data"])

    # -- 2. Feature engineering (must match training) ------------------
    logger.info("[2/4] Engineering features ...")
    target_col = cfg["competition"]["target"]
    feat_cfg = cfg["features"]
    engineer = ColorFeatureEngineer(
        drop_cols=feat_cfg["drop_cols"],
        color_pairs=[tuple(p) for p in feat_cfg["color_pairs"]],
    )
    engineer.fit(train.drop(columns=[target_col]))
    X_test = engineer.transform(test.copy())
    logger.info("  X_test: %s", X_test.shape)

    # -- 3. Load ensemble and predict ----------------------------------
    logger.info("[3/4] Loading ensemble from %s ...", model_path)
    t0 = time.time()
    ensemble = StackingEnsemble.load(model_path)
    test_preds = ensemble.predict(X_test)
    logger.info("  Predicted %d rows (%.1fs)", len(test_preds), time.time() - t0)

    # -- 4. Save submission --------------------------------------------
    output_path = args.output or str(run_dir / "submission.csv")
    logger.info("[4/4] Saving submission → %s", output_path)
    save_submission(test["id"], test_preds, output_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
