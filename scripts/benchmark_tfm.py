"""Benchmark external tabular foundation / AutoML backends.

Usage:
    uv run python scripts/benchmark_tfm.py --config config/tfm/dummy.yaml --run-name smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from stellar.blending import (
    align_proba,
    labels_from_proba,
    score_proba,
    tune_class_multipliers,
    write_probability_artifact,
)
from stellar.data import load_config, load_data
from stellar.foundation import build_feature_matrices, make_backend, stratified_context_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def _make_run_dir(output_dir: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "tfm"
    run_dir = Path(output_dir) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {"gpu": "unavailable"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        summary["gpu_error"] = str(exc)
        return summary

    if result.returncode == 0:
        summary["gpu"] = result.stdout.strip() or "available"
    else:
        summary["gpu_error"] = (result.stderr or result.stdout).strip()
    return summary


def _predict_proba_batched(backend, X, batch_size: int | None) -> np.ndarray:
    if batch_size is None or batch_size <= 0 or batch_size >= len(X):
        return backend.predict_proba(X)

    chunks = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        chunks.append(backend.predict_proba(X.iloc[start:stop]))
    return np.vstack(chunks)


def run_benchmark(config_path: str, run_name: str | None = None) -> Path:
    cfg = load_config(config_path)
    output_dir = cfg.get("paths", {}).get("tfm_outputs", "outputs/tfm")
    run_dir = _make_run_dir(output_dir, run_name)
    logger.info("TFM benchmark run: %s", run_dir)

    t0 = time.time()

    data_cfg = cfg.get("data", {})
    train, test = load_data(
        cfg.get("paths", {}).get("data", "data/"),
        train_file=data_cfg.get("train_file", "train.csv"),
        test_file=data_cfg.get("test_file", "test.csv"),
        augment_path=data_cfg.get("augment_path"),
        dedup_cols=data_cfg.get("dedup_cols"),
    )

    target_col = cfg["competition"].get("target", "class")
    class_labels = cfg["competition"].get("classes", sorted(train[target_col].unique()))
    classes = [str(c) for c in class_labels]
    y_train = train[target_col].copy()

    feature_cfg = cfg.get("features", {})
    feature_cfg = {**feature_cfg, "target_col": target_col}
    X_train, X_test = build_feature_matrices(train, test, y_train, feature_cfg)
    logger.info("Features: train=%s test=%s", X_train.shape, X_test.shape)

    cv_cfg = cfg["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_state", 42),
    )

    backend_cfg = cfg["backend"]
    context_rows = backend_cfg.get("context_rows")
    context_random_state = backend_cfg.get("context_random_state", cv_cfg.get("random_state", 42))
    predict_batch_size = backend_cfg.get("predict_batch_size")

    oof_proba = np.zeros((len(X_train), len(classes)), dtype=float)
    test_proba = np.zeros((len(X_test), len(classes)), dtype=float)
    valid_scores: list[float] = []
    context_sizes: list[int] = []

    for fold, (train_idx, val_idx) in tqdm(
        list(enumerate(cv.split(X_train, y_train))),
        desc="CV fold",
        unit="fold",
    ):
        fold_dir = run_dir / "fold_models" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        context_idx = stratified_context_indices(
            y_fold,
            context_rows,
            random_state=context_random_state + fold,
        )
        X_context = X_fold.iloc[context_idx]
        y_context = y_fold.iloc[context_idx]
        context_sizes.append(len(X_context))

        backend = make_backend(backend_cfg, fold_dir)
        backend.fit(X_context, y_context)

        val_proba = _predict_proba_batched(backend, X_train.iloc[val_idx], predict_batch_size)
        val_proba = align_proba(val_proba, backend.classes_, classes)
        oof_proba[val_idx] = val_proba
        valid_scores.append(score_proba(val_proba, y_train.iloc[val_idx], classes))

        fold_test = _predict_proba_batched(backend, X_test, predict_batch_size)
        fold_test = align_proba(fold_test, backend.classes_, classes)
        test_proba += fold_test / cv.get_n_splits()

        logger.info(
            "Fold %d: context=%d valid_bal_acc=%.4f",
            fold,
            len(X_context),
            valid_scores[-1],
        )

    tune_thresholds = cfg.get("benchmark", {}).get("tune_thresholds", False)
    threshold_multipliers = None
    if tune_thresholds:
        threshold_multipliers, overall_score = tune_class_multipliers(oof_proba, y_train, classes)
    else:
        overall_score = float(
            balanced_accuracy_score(y_train.to_numpy(), labels_from_proba(oof_proba, classes))
        )

    predictions = labels_from_proba(test_proba, classes, multipliers=threshold_multipliers)
    elapsed = time.time() - t0
    hardware = _hardware_summary()

    metrics: dict[str, Any] = {
        "overall_oof_score": round(overall_score, 6),
        "valid_scores": [round(float(s), 6) for s in valid_scores],
        "mean_valid_score": round(float(np.mean(valid_scores)), 6),
        "n_train_context": context_rows,
        "actual_context_sizes": context_sizes,
        "n_features": int(X_train.shape[1]),
        "model_family": backend_cfg["name"],
        "wall_time_seconds": round(elapsed, 2),
        "hardware": hardware,
        "data_config": feature_cfg.get("mode", "raw"),
        "classes": classes,
        "tune_thresholds": tune_thresholds,
        "threshold_multipliers": threshold_multipliers.tolist()
        if threshold_multipliers is not None
        else None,
    }

    write_probability_artifact(
        run_dir=run_dir,
        config=cfg,
        metrics=metrics,
        oof_proba=oof_proba,
        test_proba=test_proba,
        train_ids=np.arange(len(train)),
        test_ids=test["id"].to_numpy(),
        classes=classes,
        y_true=y_train.to_numpy(),
        predictions=predictions,
    )

    with open(run_dir / "benchmark_summary.json", "w") as f:
        json.dump({"run_dir": str(run_dir), **metrics}, f, indent=2, default=str)

    logger.info("Saved artifact to %s", run_dir)
    logger.info("OOF balanced accuracy: %.4f", overall_score)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a tabular foundation backend.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    parser.add_argument("--run-name", default=None, help="Human-readable run suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_benchmark(args.config, args.run_name)
    print(run_dir)


if __name__ == "__main__":
    main()
