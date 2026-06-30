"""Blend saved probability artifacts.

Usage:
    uv run python scripts/blend_predictions.py \
        --runs outputs/tfm/run_a outputs/tfm/run_b \
        --run-name blend_a_b
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stellar.blending import (
    align_proba,
    blend_probas,
    labels_from_proba,
    optimize_blend_weights,
    read_probability_artifact,
    score_proba,
    tune_class_multipliers,
    write_probability_artifact,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def _make_run_dir(output_dir: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "blend"
    run_dir = Path(output_dir) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _assert_same_ids(name: str, arrays: list[np.ndarray]) -> None:
    first = arrays[0]
    for idx, arr in enumerate(arrays[1:], start=1):
        if not np.array_equal(first, arr, equal_nan=True):
            raise ValueError(f"{name} mismatch between artifact 0 and artifact {idx}")


def run_blend(
    run_paths: list[str],
    run_name: str | None = None,
    output_dir: str = "outputs/tfm",
    per_class: bool = False,
    tune_thresholds: bool = True,
) -> Path:
    artifacts = [read_probability_artifact(path) for path in run_paths]
    if len(artifacts) < 2:
        raise ValueError("At least two artifacts are required for blending")

    classes = artifacts[0].classes
    if artifacts[0].y_true is None:
        raise ValueError(f"Artifact {artifacts[0].run_dir} does not include y_true.npy")
    y_true = artifacts[0].y_true

    _assert_same_ids("train_ids", [a.train_ids for a in artifacts])
    _assert_same_ids("test_ids", [a.test_ids for a in artifacts])
    for artifact in artifacts[1:]:
        if artifact.y_true is None:
            raise ValueError(f"Artifact {artifact.run_dir} does not include y_true.npy")
        if not np.array_equal(y_true, artifact.y_true):
            raise ValueError(f"y_true mismatch for {artifact.run_dir}")

    oof_probas = [align_proba(a.oof_proba, a.classes, classes) for a in artifacts]
    test_probas = [align_proba(a.test_proba, a.classes, classes) for a in artifacts]

    global_weights, global_score = optimize_blend_weights(
        oof_probas,
        y_true,
        classes,
        per_class=False,
    )
    selected_kind = "global"
    selected_weights = global_weights
    selected_score = global_score

    per_class_weights = None
    per_class_score = None
    if per_class:
        per_class_weights, per_class_score = optimize_blend_weights(
            oof_probas,
            y_true,
            classes,
            per_class=True,
        )
        if per_class_score >= selected_score:
            selected_kind = "per_class"
            selected_weights = per_class_weights
            selected_score = per_class_score

    blended_oof = blend_probas(oof_probas, selected_weights)
    blended_test = blend_probas(test_probas, selected_weights)

    threshold_multipliers = None
    tuned_score = selected_score
    if tune_thresholds:
        threshold_multipliers, tuned_score = tune_class_multipliers(blended_oof, y_true, classes)

    predictions = labels_from_proba(
        blended_test,
        classes,
        multipliers=threshold_multipliers,
    )

    run_dir = _make_run_dir(output_dir, run_name)
    config: dict[str, Any] = {
        "blend": {
            "runs": [str(a.run_dir) for a in artifacts],
            "selected_weight_type": selected_kind,
            "per_class_requested": per_class,
            "tune_thresholds": tune_thresholds,
        }
    }
    metrics: dict[str, Any] = {
        "overall_oof_score": round(float(tuned_score), 6),
        "pre_threshold_oof_score": round(float(score_proba(blended_oof, y_true, classes)), 6),
        "global_blend_score": round(float(global_score), 6),
        "per_class_blend_score": round(float(per_class_score), 6)
        if per_class_score is not None
        else None,
        "selected_weight_type": selected_kind,
        "weights": selected_weights.tolist(),
        "threshold_multipliers": threshold_multipliers.tolist()
        if threshold_multipliers is not None
        else None,
        "model_family": "blend",
        "classes": classes,
        "source_runs": [str(a.run_dir) for a in artifacts],
    }

    write_probability_artifact(
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        oof_proba=blended_oof,
        test_proba=blended_test,
        train_ids=artifacts[0].train_ids,
        test_ids=artifacts[0].test_ids,
        classes=classes,
        y_true=y_true,
        predictions=predictions,
    )

    with open(run_dir / "blend_summary.json", "w") as f:
        json.dump({"run_dir": str(run_dir), **metrics}, f, indent=2, default=str)

    logger.info("Saved blend artifact to %s", run_dir)
    logger.info("OOF balanced accuracy: %.4f", tuned_score)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend saved probability artifacts.")
    parser.add_argument("--runs", nargs="+", required=True, help="Artifact directories to blend")
    parser.add_argument("--run-name", default=None, help="Human-readable run suffix")
    parser.add_argument("--output-dir", default="outputs/tfm", help="Directory for blend artifacts")
    parser.add_argument("--per-class", action="store_true", help="Try per-class blend weights")
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="Disable post-blend per-class threshold tuning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_blend(
        run_paths=args.runs,
        run_name=args.run_name,
        output_dir=args.output_dir,
        per_class=args.per_class,
        tune_thresholds=not args.no_thresholds,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
