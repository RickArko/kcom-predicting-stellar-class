"""Compare all experiment runs in a table.

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py --sort-by overall_oof_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experiment runs.")
    parser.add_argument("--runs-dir", type=str, default="outputs/runs")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="overall_oof_score",
        help="Column to sort results by",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}")
        return

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        meta = data.get("metrics", {})
        params = data.get("params", {})
        submission_path = run_dir / "submission.csv"
        rows.append(
            {
                "run": run_dir.name,
                "overall_oof_score": meta.get("overall_oof_score"),
                "mean_valid_score": (
                    round(sum(meta.get("valid_scores", [])) / len(meta["valid_scores"]), 4)
                    if meta.get("valid_scores")
                    else None
                ),
                "n_features": meta.get("n_features"),
                "n_base_models": params.get("n_base_models"),
                "cv_n_splits": params.get("cv_n_splits"),
                "elapsed_seconds": round(data.get("elapsed_seconds", 0), 1),
                "submission_exists": submission_path.exists(),
            }
        )

    if not rows:
        print("No experiment runs found.")
        return

    df = pd.DataFrame(rows)
    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.colheader_justify", "right")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
