"""Select the highest-OOF local submission artifact.

Usage:
    uv run python scripts/select_best_submission.py
    uv run python scripts/select_best_submission.py --field path
    uv run python scripts/select_best_submission.py --field message
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _score_from_training_metrics(data: dict[str, Any]) -> float | None:
    return data.get("metrics", {}).get("overall_oof_score")


def _score_from_tfm_metrics(data: dict[str, Any]) -> float | None:
    return data.get("overall_oof_score")


def _collect_runs(base_dir: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    if not base_dir.exists():
        return rows

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        submission_path = run_dir / "submission.csv"
        if not metrics_path.exists() or not submission_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        if source == "tfm":
            score = _score_from_tfm_metrics(data)
        else:
            score = _score_from_training_metrics(data)
        if score is None:
            continue

        rows.append(
            {
                "source": source,
                "run": run_dir.name,
                "score": float(score),
                "path": str(submission_path),
            }
        )
    return rows


def select_best(
    runs_dir: str | Path = "outputs/runs",
    tfm_dir: str | Path = "outputs/tfm",
) -> dict[str, Any]:
    rows = _collect_runs(Path(runs_dir), "run")
    rows.extend(_collect_runs(Path(tfm_dir), "tfm"))
    if not rows:
        raise FileNotFoundError("No scored local submissions found in outputs/runs or outputs/tfm")
    return sorted(rows, key=lambda r: (r["score"], r["run"]), reverse=True)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best local submission by OOF score.")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--tfm-dir", default="outputs/tfm")
    parser.add_argument(
        "--field",
        choices=["path", "message", "json"],
        default="json",
        help="Output format for shell automation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = select_best(args.runs_dir, args.tfm_dir)
    if args.field == "path":
        print(best["path"])
    elif args.field == "message":
        print(f"best local OOF: {best['source']} {best['run']} OOF {best['score']:.4f}")
    else:
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
