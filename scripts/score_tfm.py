"""Rank saved TFM/blend probability artifacts.

Usage:
    uv run python scripts/score_tfm.py
    uv run python scripts/score_tfm.py --base-dir outputs/tfm --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_scores(base_dir: str | Path = "outputs/tfm") -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(Path(base_dir).glob("*/metrics.json")):
        run_dir = metrics_path.parent
        with open(metrics_path) as f:
            metrics = json.load(f)

        rows.append(
            {
                "run": run_dir.name,
                "model_family": metrics.get("model_family"),
                "overall_oof_score": metrics.get("overall_oof_score"),
                "mean_valid_score": metrics.get("mean_valid_score"),
                "n_features": metrics.get("n_features"),
                "n_train_context": metrics.get("n_train_context"),
                "wall_time_seconds": metrics.get("wall_time_seconds"),
                "submission_exists": (run_dir / "submission.csv").exists(),
                "path": str(run_dir),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run",
                "model_family",
                "overall_oof_score",
                "mean_valid_score",
                "n_features",
                "n_train_context",
                "wall_time_seconds",
                "submission_exists",
                "path",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("overall_oof_score", ascending=False, na_position="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank TFM/blend artifacts by OOF score.")
    parser.add_argument("--base-dir", default="outputs/tfm", help="Artifact directory to scan")
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_scores(args.base_dir).head(args.top)
    if df.empty:
        print(f"No TFM artifacts found under {args.base_dir}")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
