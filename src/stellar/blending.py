from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score

from stellar.models import save_submission


@dataclass
class ProbabilityArtifact:
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    oof_proba: np.ndarray
    test_proba: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    classes: list[str]
    y_true: np.ndarray | None = None


def align_proba(
    proba: np.ndarray,
    source_classes: list[str] | np.ndarray,
    target_classes: list[str] | np.ndarray,
) -> np.ndarray:
    """Reorder probability columns from source class order to target class order."""
    source = [str(c) for c in source_classes]
    target = [str(c) for c in target_classes]
    missing = sorted(set(target) - set(source))
    if missing:
        raise ValueError(f"Missing probability columns for classes: {missing}")

    out = np.zeros((proba.shape[0], len(target)), dtype=float)
    for target_idx, label in enumerate(target):
        out[:, target_idx] = proba[:, source.index(label)]
    return out


def labels_from_proba(
    proba: np.ndarray,
    classes: list[str] | np.ndarray,
    multipliers: np.ndarray | None = None,
) -> np.ndarray:
    scores = proba if multipliers is None else proba * multipliers[np.newaxis, :]
    class_arr = np.asarray(classes)
    return class_arr[np.argmax(scores, axis=1)]


def score_proba(
    proba: np.ndarray,
    y_true: np.ndarray | pd.Series,
    classes: list[str] | np.ndarray,
    multipliers: np.ndarray | None = None,
) -> float:
    preds = labels_from_proba(proba, classes, multipliers=multipliers)
    return float(balanced_accuracy_score(np.asarray(y_true), preds))


def tune_class_multipliers(
    proba: np.ndarray,
    y_true: np.ndarray | pd.Series,
    classes: list[str] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimize per-class score multipliers against OOF balanced accuracy."""
    from scipy.optimize import minimize

    y_arr = np.asarray(y_true)

    def objective(raw: np.ndarray) -> float:
        multipliers = np.exp(raw)
        return -score_proba(proba, y_arr, classes, multipliers=multipliers)

    result = minimize(objective, np.zeros(proba.shape[1]), method="Nelder-Mead")
    multipliers = np.exp(result.x)
    return multipliers, -float(result.fun)


def blend_probas(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Blend probabilities with global or per-class weights."""
    stacked = np.stack(probas, axis=0)
    if weights.ndim == 1:
        return np.tensordot(weights, stacked, axes=(0, 0))
    if weights.shape != (len(probas), probas[0].shape[1]):
        raise ValueError(
            "Per-class weights must have shape "
            f"({len(probas)}, {probas[0].shape[1]}), got {weights.shape}"
        )
    return (stacked * weights[:, np.newaxis, :]).sum(axis=0)


def optimize_blend_weights(
    oof_probas: list[np.ndarray],
    y_true: np.ndarray | pd.Series,
    classes: list[str] | np.ndarray,
    per_class: bool = False,
) -> tuple[np.ndarray, float]:
    """Optimize non-negative blend weights on OOF balanced accuracy."""
    from scipy.optimize import minimize

    if len(oof_probas) == 1:
        weights = np.ones((1, oof_probas[0].shape[1])) if per_class else np.ones(1)
        return weights, score_proba(oof_probas[0], y_true, classes)

    n_models = len(oof_probas)
    n_classes = oof_probas[0].shape[1]
    y_arr = np.asarray(y_true)

    if per_class:
        x0 = np.full(n_models * n_classes, 1.0 / n_models)
        bounds = [(0.0, 1.0)] * len(x0)
        constraints = [
            {
                "type": "eq",
                "fun": lambda x, class_idx=class_idx: (
                    x.reshape(n_models, n_classes)[:, class_idx].sum() - 1.0
                ),
            }
            for class_idx in range(n_classes)
        ]

        def objective(x: np.ndarray) -> float:
            weights = x.reshape(n_models, n_classes)
            proba = blend_probas(oof_probas, weights)
            return -score_proba(proba, y_arr, classes)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x.reshape(n_models, n_classes)
    else:
        x0 = np.full(n_models, 1.0 / n_models)
        bounds = [(0.0, 1.0)] * n_models
        constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]

        def objective(x: np.ndarray) -> float:
            proba = blend_probas(oof_probas, x)
            return -score_proba(proba, y_arr, classes)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x

    blended = blend_probas(oof_probas, weights)
    return weights, score_proba(blended, y_arr, classes)


def write_probability_artifact(
    run_dir: str | Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    train_ids: np.ndarray | pd.Series,
    test_ids: np.ndarray | pd.Series,
    classes: list[str] | np.ndarray,
    y_true: np.ndarray | pd.Series | None = None,
    predictions: np.ndarray | pd.Series | None = None,
) -> None:
    """Write the standard TFM/blend artifact directory."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    classes_list = [str(c) for c in classes]
    with open(path / "classes.json", "w") as f:
        json.dump(classes_list, f, indent=2)

    np.save(path / "oof_proba.npy", oof_proba)
    np.save(path / "test_proba.npy", test_proba)
    np.save(path / "train_ids.npy", np.asarray(train_ids))
    np.save(path / "test_ids.npy", np.asarray(test_ids))
    if y_true is not None:
        np.save(path / "y_true.npy", np.asarray(y_true))
    if predictions is not None:
        save_submission(pd.Series(test_ids), np.asarray(predictions), str(path / "submission.csv"))


def read_probability_artifact(run_dir: str | Path) -> ProbabilityArtifact:
    """Load a standard TFM/blend probability artifact directory."""
    path = Path(run_dir)
    required = [
        "config.yaml",
        "metrics.json",
        "classes.json",
        "oof_proba.npy",
        "test_proba.npy",
        "train_ids.npy",
        "test_ids.npy",
    ]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing artifact files: {missing}")

    with open(path / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(path / "metrics.json") as f:
        metrics = json.load(f)
    with open(path / "classes.json") as f:
        classes = json.load(f)

    y_path = path / "y_true.npy"
    y_true = np.load(y_path, allow_pickle=True) if y_path.exists() else None

    return ProbabilityArtifact(
        run_dir=path,
        config=config,
        metrics=metrics,
        oof_proba=np.load(path / "oof_proba.npy", allow_pickle=False),
        test_proba=np.load(path / "test_proba.npy", allow_pickle=False),
        train_ids=np.load(path / "train_ids.npy", allow_pickle=True),
        test_ids=np.load(path / "test_ids.npy", allow_pickle=True),
        classes=[str(c) for c in classes],
        y_true=y_true,
    )
