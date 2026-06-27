"""Adversarial validation — detect distribution shift between train and test.

Usage:
    uv run python scripts/adversarial_validate.py
    uv run python scripts/adversarial_validate.py --config config/config.yaml

If the AUC is > 0.8 there is significant distribution shift.  The per-feature
importance table shows which columns contribute most to the separation.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from stellar.data import load_config, load_data
from stellar.features import ColorFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial validation.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Adversarial validation")
    logger.info("=" * 60)

    data_cfg = cfg.get("data", {})
    train, test = load_data(
        cfg["paths"]["data"],
        augment_path=data_cfg.get("augment_path"),
        dedup_cols=data_cfg.get("dedup_cols"),
    )

    target_col = cfg["competition"]["target"]
    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col], errors="ignore")
    X_test = test.copy()

    feat_cfg = cfg["features"]
    engineer = ColorFeatureEngineer(
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
    X_train = engineer.fit_transform(X_train, y_train)
    X_test = engineer.transform(X_test)

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X_adv = pd.concat(
        [X_train, X_test],
        axis=0,
        keys=["train", "test"],
        names=["source"],
    ).reset_index(drop=True)
    y_adv = np.zeros(len(X_adv), dtype=int)
    y_adv[len(X_train) :] = 1

    logger.info("Train set: %d  Test set: %d", len(X_train), len(X_test))

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X_adv, y_adv, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_adv, preds)
    logger.info("Adversarial validation AUC: %.4f", auc)

    if auc > 0.8:
        logger.warning("Significant distribution shift detected (AUC > 0.8)!")

    model.fit(X_adv, y_adv)
    importances = pd.DataFrame(
        {
            "feature": X_adv.columns,
            "importance": model.feature_importances_,
        },
    ).sort_values("importance", ascending=False)

    print("\nTop-20 features by adversarial importance:\n")
    print(importances.head(20).to_string(index=False))

    out_path = "outputs/adversarial_results.csv"
    importances.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)
    logger.info("Adversarial validation complete — AUC: %.4f", auc)


if __name__ == "__main__":
    main()
