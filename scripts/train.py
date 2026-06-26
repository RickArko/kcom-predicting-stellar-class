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

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from stellar.data import load_config, load_data
from stellar.features import ColorFeatureEngineer
from stellar.models import SimpleAverageMeta, StackingEnsemble, save_submission
from stellar.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

_SEED_PARAMS = {
    "lgbm": "random_state",
    "xgb": "random_state",
    "catboost": "random_seed",
}


def _expand_seeds(
    cfg: dict,
    model_cls: type,
    seed_param: str,
    name_prefix: str,
) -> list[tuple[str, object]]:
    """Expand a model config into multiple instances with different seeds.

    If the config contains a ``seeds`` list each seed produces one instance.
    If absent, the single value of *seed_param* is used (default 42).
    """
    seeds = cfg.pop("seeds", [cfg.get(seed_param, 42)])
    cfg.pop(seed_param, None)
    models: list[tuple[str, object]] = []
    for i, seed in enumerate(seeds):
        instance_cfg = cfg.copy()
        instance_cfg[seed_param] = seed
        suffix = f"_s{i}" if len(seeds) > 1 else ""
        models.append((f"{name_prefix}{suffix}", model_cls(**instance_cfg)))
    return models


def _build_ensemble(cfg: dict) -> tuple[list[tuple[str, object]], object, dict[str, dict]]:
    """Build base-model list and meta-model from config."""
    model_cfgs = {}
    for name in ("lgbm", "xgb", "catboost"):
        model_cfgs[name] = cfg[name].copy()

    catboost_fit_kwargs = {}
    if "cat_features" in model_cfgs["catboost"]:
        catboost_fit_kwargs["cat_features"] = model_cfgs["catboost"].pop("cat_features")

    base_models: list[tuple[str, object]] = []
    base_models.extend(
        _expand_seeds(model_cfgs["lgbm"], LGBMClassifier, _SEED_PARAMS["lgbm"], "lgbm"),
    )
    base_models.extend(
        _expand_seeds(model_cfgs["xgb"], XGBClassifier, _SEED_PARAMS["xgb"], "xgb"),
    )
    base_models.extend(
        _expand_seeds(
            model_cfgs["catboost"],
            CatBoostClassifier,
            _SEED_PARAMS["catboost"],
            "catboost",
        ),
    )

    meta_params = cfg.get("meta", {}).copy()
    meta_type = meta_params.pop("model", "logistic_regression")
    calibrated = meta_params.pop("calibrated", False)

    if meta_type == "simple_average":
        meta_model: object = SimpleAverageMeta()
    else:
        meta_params.setdefault("max_iter", 1000)
        meta_params.setdefault("random_state", 42)
        lr = LogisticRegression(**meta_params)
        if calibrated:
            from sklearn.calibration import CalibratedClassifierCV

            meta_model = CalibratedClassifierCV(lr, cv=3, method="sigmoid")
        else:
            meta_model = lr

    model_fit_kwargs: dict[str, dict] = {}
    if catboost_fit_kwargs:
        for name, _ in base_models:
            if name.startswith("catboost"):
                model_fit_kwargs[name] = catboost_fit_kwargs

    return base_models, meta_model, model_fit_kwargs


def _build_engineer(feat_cfg: dict) -> ColorFeatureEngineer:
    """Build feature engineer from the features section of config."""
    return ColorFeatureEngineer(
        drop_cols=feat_cfg["drop_cols"],
        color_pairs=[tuple(p) for p in feat_cfg["color_pairs"]],
        cat_cols=feat_cfg.get("cat_cols"),
        encoding=feat_cfg.get("encoding", "ohe"),
        interaction_pairs=[tuple(p) for p in feat_cfg.get("interaction_pairs", [])],
        ratio_pairs=[tuple(p) for p in feat_cfg.get("ratio_pairs", [])],
        log_transform_cols=feat_cfg.get("log_transform_cols"),
        poly_cols=feat_cfg.get("poly_cols"),
        polynomial_degree=feat_cfg.get("polynomial_degree"),
    )


def _run_pseudo_labeling(
    base_models: list[tuple[str, object]],
    meta_model: object,
    model_fit_kwargs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
    pseudo_cfg: dict,
) -> tuple[StackingEnsemble, pd.DataFrame, pd.Series]:
    """Iterative pseudo-labeling: add confident test predictions to training set."""
    threshold = pseudo_cfg["confidence_threshold"]
    max_iterations = pseudo_cfg.get("max_iterations", 1)

    X_cur = X_train.copy()
    y_cur = y_train.copy()

    ensemble: StackingEnsemble | None = None
    for iteration in range(max_iterations):
        logger.info("  Pseudo-label iteration %d/%d ...", iteration + 1, max_iterations)
        ensemble = StackingEnsemble(base_models, meta_model)
        ensemble.fit(X_cur, y_cur, cv, X_test, model_fit_kwargs=model_fit_kwargs)

        probas = ensemble.predict_proba(X_test)
        max_probas = np.max(probas, axis=1)
        confident = max_probas >= threshold

        if confident.sum() == 0:
            logger.info("    No confident predictions above %.2f — stopping.", threshold)
            break

        le = ensemble.label_encoder_
        confident_preds = le.inverse_transform(np.argmax(probas[confident], axis=1))
        logger.info("    Adding %d confident test rows.", confident.sum())

        X_cur = pd.concat([X_cur, X_test.iloc[confident]], ignore_index=True)
        y_cur = pd.concat(
            [y_cur, pd.Series(confident_preds, name=y_train.name)],
            ignore_index=True,
        )

    assert ensemble is not None
    return ensemble, X_cur, y_cur


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
    data_cfg = cfg.get("data", {})
    train, test = load_data(
        cfg["paths"]["data"],
        augment_path=data_cfg.get("augment_path"),
        dedup_cols=data_cfg.get("dedup_cols"),
    )
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
    engineer = _build_engineer(feat_cfg)
    need_y_for_fe = engineer.encoding == "target"
    X_train = engineer.fit_transform(X_train, y_train if need_y_for_fe else None)
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
    base_models, meta_model, model_fit_kwargs = _build_ensemble(cfg)
    meta_type = cfg.get("meta", {}).get("model", "logistic_regression")
    tune_thresholds = cfg.get("meta", {}).get("tune_thresholds", False)

    # -- 6. Train with tracking ----------------------------------------
    logger.info(
        "[3/5] Training with %d-fold CV (%d base models) ...", cv_cfg["n_splits"], len(base_models)
    )
    t0 = time.time()

    with track_experiment(cfg, run_name=args.run_name) as run:
        pseudo_cfg = cfg.get("pseudo_label", {})
        if pseudo_cfg.get("enabled", False):
            ensemble, _, _ = _run_pseudo_labeling(
                base_models,
                meta_model,
                model_fit_kwargs,
                X_train,
                y_train,
                X_test,
                cv,
                pseudo_cfg,
            )
        else:
            ensemble = StackingEnsemble(base_models, meta_model)
            ensemble.fit(X_train, y_train, cv, X_test, model_fit_kwargs=model_fit_kwargs)

        if tune_thresholds:
            tuned_score = ensemble.tune_thresholds()
            logger.info("  Threshold-tuned OOF balanced accuracy: %.4f", tuned_score)

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
                "meta_model": meta_type,
                "tune_thresholds": tune_thresholds,
            }
        )

        # Save trained ensemble
        model_path = run.models_dir / "ensemble.joblib"
        ensemble.save(model_path)
        logger.info("  Model saved to %s", model_path)

        # Generate & save submission (run-specific + canonical)
        logger.info("[4/5] Generating predictions ...")
        if tune_thresholds:
            test_preds = ensemble.predict_with_thresholds(X_test)
        else:
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
