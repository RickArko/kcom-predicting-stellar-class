"""End-to-end training pipeline for Predicting Stellar Class.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config config/config.yaml --run-name expr-001
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from stellar.data import load_config, load_data
from stellar.features import ColorFeatureEngineer
from stellar.models import StackingEnsemble, save_submission
from stellar.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stellar classification ensemble.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Human-readable experiment name (appended to timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Predicting Stellar Class — Training Pipeline")
    logger.info("=" * 60)

    # -- 1. Load data --------------------------------------------------
    logger.info("[1/5] Loading data ...")
    t0 = time.time()
    train, test = load_data(cfg["paths"]["data"])
    logger.info("  Train: %s  Test: %s  (%.1fs)", train.shape, test.shape, time.time() - t0)

    # -- 2. Separate target from features ------------------------------
    target_col = cfg["competition"]["target"]
    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col])
    X_test = test.copy()

    # -- 3. Feature engineering ----------------------------------------
    logger.info("[2/5] Engineering features ...")
    t0 = time.time()
    feat_cfg = cfg["features"]
    engineer = ColorFeatureEngineer(
        drop_cols=feat_cfg["drop_cols"],
        color_pairs=[tuple(p) for p in feat_cfg["color_pairs"]],
        cat_cols=feat_cfg.get("cat_cols"),
        encoding=feat_cfg.get("encoding", "ohe"),
    )
    X_train = engineer.fit_transform(X_train)
    X_test = engineer.transform(X_test)
    logger.info("  X_train: %s  X_test: %s  (%.1fs)", X_train.shape, X_test.shape, time.time() - t0)

    # -- 4. Cross-validation setup -------------------------------------
    cv_cfg = cfg["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )

    # -- 5. Build ensemble ---------------------------------------------
    lgbm_cfg = cfg["lgbm"].copy()
    xgb_cfg = cfg["xgb"].copy()
    catboost_cfg = cfg["catboost"].copy()

    catboost_fit_kwargs = {}
    if "cat_features" in catboost_cfg:
        catboost_fit_kwargs["cat_features"] = catboost_cfg.pop("cat_features")

    base_models = [
        ("lgbm", LGBMClassifier(**lgbm_cfg)),
        ("xgb", XGBClassifier(**xgb_cfg)),
        ("catboost", CatBoostClassifier(**catboost_cfg)),
    ]
    meta_params = cfg.get("meta", {"max_iter": 1000, "random_state": 42})
    meta_model = LogisticRegression(**meta_params)
    ensemble = StackingEnsemble(base_models, meta_model)

    model_fit_kwargs: dict[str, dict] = {}
    if catboost_fit_kwargs:
        model_fit_kwargs["catboost"] = catboost_fit_kwargs

    # -- 6. Train with tracking ----------------------------------------
    logger.info("[3/5] Training with %d-fold CV ...", cv_cfg["n_splits"])
    t0 = time.time()

    with track_experiment(cfg, run_name=args.run_name) as run:
        ensemble.fit(X_train, y_train, cv, X_test, model_fit_kwargs=model_fit_kwargs)

        run.log_metrics(
            {
                "valid_scores": [round(s, 4) for s in ensemble.valid_scores_],
                "overall_oof_score": round(ensemble.overall_oof_score_, 4),
                "n_features": X_train.shape[1],
            }
        )
        run.log_params(
            {
                "cv_n_splits": cv_cfg["n_splits"],
                "n_base_models": len(base_models),
            }
        )

        # Save trained ensemble
        model_path = run.models_dir / "ensemble.joblib"
        ensemble.save(model_path)
        logger.info("  Model saved to %s", model_path)

        # Generate & save submission (run-specific + canonical)
        logger.info("[4/5] Generating predictions ...")
        test_preds = ensemble.predict(X_test)
        save_submission(test["id"], test_preds, str(run.submission_path))
        logger.info("  Submission saved to %s", run.submission_path)

        canonical_sub = Path(cfg["paths"]["submissions"]) / "submission.csv"
        save_submission(test["id"], test_preds, str(canonical_sub))
        logger.info("  Canonical submission → %s", canonical_sub)

    elapsed = time.time() - t0
    logger.info("[5/5] Done in %.1fs", elapsed)
    logger.info("  OOF balanced accuracy (stacked): %.4f", ensemble.overall_oof_score_)


if __name__ == "__main__":
    main()
