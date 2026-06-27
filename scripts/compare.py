"""Compare all experiment runs in a table.

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py --sort-by overall_oof_score
    uv run python scripts/compare.py --feature-importance
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
    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Run permutation importance on the best run's OOF meta-features",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to show (default 20)",
    )
    return parser.parse_args()


def _find_best_run(runs_dir: Path) -> Path | None:
    best_score = -1.0
    best_run = None
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        score = data.get("metrics", {}).get("overall_oof_score", -1.0)
        if score > best_score:
            best_score = score
            best_run = run_dir
    return best_run


def _run_feature_importance(runs_dir: Path, top_n: int) -> None:
    from sklearn.inspection import permutation_importance

    from stellar.models import StackingEnsemble

    best_run = _find_best_run(runs_dir)
    if best_run is None:
        print("No runs found.")
        return

    model_path = best_run / "models" / "ensemble.joblib"
    if not model_path.exists():
        print(f"No ensemble found in {best_run.name}")
        return

    print(f"Loading ensemble from {best_run.name} ...")
    ensemble = StackingEnsemble.load(str(model_path))

    feature_names = []
    for name, _ in ensemble.base_models:
        for cls in ensemble.label_encoder_.classes_:
            feature_names.append(f"{name}_{cls}")

    print("Running permutation importance (10 repeats) ...")
    result = permutation_importance(
        ensemble.meta_model_,
        ensemble.oof_meta_,
        ensemble.y_enc_,
        n_repeats=10,
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    feat_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
    ).sort_values("importance_mean", ascending=False)

    print(f"\nTop-{top_n} features by permutation importance:\n")
    print(feat_df.head(top_n).to_string(index=False))

    feat_df.head(top_n).to_csv(best_run / "feature_importance.csv", index=False)
    print(f"\nSaved to {best_run / 'feature_importance.csv'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        display = feat_df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
        means = display["importance_mean"].values
        stds = display["importance_std"].values
        ax.barh(range(len(display)), means, xerr=stds)
        ax.set_yticks(range(len(display)))
        ax.set_yticklabels(display["feature"].values)
        ax.set_xlabel("Permutation importance (balanced accuracy drop)")
        ax.set_title(f"Feature importance — {best_run.name}")
        fig.tight_layout()
        plot_path = best_run / "feature_importance.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}")
        return

    if args.feature_importance:
        _run_feature_importance(runs_dir, args.top_n)
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
