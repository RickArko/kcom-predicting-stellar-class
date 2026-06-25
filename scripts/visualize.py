"""Visualize submission performance: CV (OOF) vs public/private leaderboard.

Reads Kaggle submission scores and local run metrics, then generates a
grouped bar chart.  Requires matplotlib (dev dependency).

Usage:
    uv run python scripts/visualize.py
    uv run python scripts/visualize.py --output docs/figures/submission_scores.png
    uv run python scripts/visualize.py --no-kaggle   # local OOF scores only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "docs/figures/submission_scores.png"
COMPETITION = "playground-series-s6e6"
TOKEN_FILE = ".kaggle/access_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize submission performance.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="outputs/runs",
        help="Directory of local experiment runs",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Skip Kaggle API call; use local OOF scores only",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default=COMPETITION,
        help="Kaggle competition slug",
    )
    return parser.parse_args()


def _get_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        p = Path(TOKEN_FILE)
        if p.exists():
            token = p.read_text().strip()
    return token


def fetch_kaggle_submissions(competition: str) -> list[dict]:
    """Fetch submission history from Kaggle via the CLI CSV output."""
    import subprocess

    token = _get_token()
    if not token:
        logger.warning("No Kaggle token found; skipping leaderboard scores.")
        return []

    env = {**os.environ, "KAGGLE_API_TOKEN": token}
    result = subprocess.run(
        [
            "uv",
            "run",
            "kaggle",
            "competitions",
            "submissions",
            "-c",
            competition,
            "-v",
            "-q",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if result.returncode != 0:
        logger.warning("Kaggle API call failed: %s", result.stderr.strip())
        return []

    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = []
    for row in reader:
        rows.append(
            {
                "ref": row.get("ref", ""),
                "description": row.get("description", ""),
                "date": row.get("date", ""),
                "public_score": _parse_score(row.get("publicScore", "")),
                "private_score": _parse_score(row.get("privateScore", "")),
            }
        )
    return rows


def _parse_score(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_local_runs(runs_dir: str) -> dict[str, dict]:
    """Load OOF scores from local run directories.

    Returns a dict keyed by run name (the part after the timestamp).
    """
    runs = {}
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return runs
    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        meta = data.get("metrics", {})
        oof = meta.get("overall_oof_score")
        if oof is None:
            continue
        name = re.sub(r"^\d{8}_\d{6}_", "", run_dir.name)
        runs[name] = {
            "oof": oof,
            "valid_scores": meta.get("valid_scores", []),
            "elapsed": data.get("elapsed_seconds", 0),
            "dir": str(run_dir),
        }
    return runs


def _short_label(name: str) -> str:
    """Shorten run names for chart labels."""
    name = name.replace("_keep_categoricals", "_cats")
    name = name.replace("_tuned_hyperparams", "_tuned")
    name = name.replace("_label_encoding", "_label")
    name = name.replace("_cb_native", "_cb")
    name = name.replace("_lgb_native", "_lgb")
    name = name.replace("_interactions", "_inter")
    name = name.replace("_no_augment", "_noaug")
    name = name.replace("_simple_avg", "_avg")
    name = name.replace("_thresholds", "_thresh")
    name = name.replace("v00", "v")
    return name


def _match_run_to_oof(description: str, runs: dict[str, dict]) -> float | None:
    """Try to find a local OOF score matching a Kaggle submission description."""
    for name, info in runs.items():
        if name in description or description.startswith(name):
            return info["oof"]
    if "OOF" in description:
        m = re.search(r"OOF\s+([\d.]+)", description)
        if m:
            return float(m.group(1))
    return None


def build_chart_data(
    runs: dict[str, dict],
    submissions: list[dict],
    max_bars: int = 10,
) -> list[dict]:
    """Merge local runs and Kaggle submissions into chart rows.

    Strategy: start with Kaggle submissions (which have real LB scores),
    dedup by description (keep latest), attach OOF from local runs.  If no
    Kaggle submissions, fall back to local runs only.
    """
    if submissions:
        seen = {}
        for s in submissions:
            desc = s["description"]
            seen[desc] = s
        rows = list(seen.values())
        rows.sort(key=lambda r: r["date"], reverse=True)
        chart = []
        for s in rows:
            oof = _match_run_to_oof(s["description"], runs)
            chart.append(
                {
                    "label": _short_label(s["description"].split(":")[0].strip()),
                    "oof": oof,
                    "public": s["public_score"],
                    "private": s["private_score"],
                }
            )
        return chart[:max_bars]

    chart = []
    for name, info in sorted(runs.items()):
        chart.append(
            {
                "label": _short_label(name),
                "oof": info["oof"],
                "public": None,
                "private": None,
            }
        )
    return chart[:max_bars]


def render_chart(chart: list[dict], output: str) -> None:
    """Render a grouped bar chart of OOF / public / private scores."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = [r["label"] for r in chart]
    n = len(chart)
    x = np.arange(n)
    width = 0.25

    has_oof = any(r["oof"] is not None for r in chart)
    has_public = any(r["public"] is not None for r in chart)
    has_private = any(r["private"] is not None for r in chart)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))

    bars_added = []
    offset = 0
    if has_oof:
        vals = [r["oof"] or 0 for r in chart]
        bar = ax.bar(x + offset * width, vals, width, label="CV (OOF)", color="#4C72B0")
        bars_added.append(bar)
        offset += 1
    if has_public:
        vals = [r["public"] or 0 for r in chart]
        bar = ax.bar(x + offset * width, vals, width, label="Public LB", color="#55A868")
        bars_added.append(bar)
        offset += 1
    if has_private:
        vals = [r["private"] or 0 for r in chart]
        bar = ax.bar(x + offset * width, vals, width, label="Private LB", color="#C44E52")
        bars_added.append(bar)
        offset += 1

    all_vals = []
    for r in chart:
        for k in ("oof", "public", "private"):
            if r[k] is not None:
                all_vals.append(r[k])
    if all_vals:
        ymin = min(all_vals) - 0.005
        ymax = max(all_vals) + 0.005
        ax.set_ylim(ymin, ymax)

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Submission Performance — CV vs Leaderboard")
    ax.set_xticks(x + width * (offset - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_added:
        for rect in bar:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Chart saved to %s", out)


def main() -> None:
    args = parse_args()

    logger.info("Loading local runs from %s ...", args.runs_dir)
    runs = load_local_runs(args.runs_dir)
    logger.info("  Found %d runs with OOF scores", len(runs))

    submissions = []
    if not args.no_kaggle:
        logger.info("Fetching Kaggle submissions ...")
        submissions = fetch_kaggle_submissions(args.competition)
        logger.info("  Found %d submissions", len(submissions))

    chart = build_chart_data(runs, submissions)
    if not chart:
        logger.error("No data to plot. Run some experiments first.")
        return

    logger.info("Rendering chart with %d bars ...", len(chart))
    render_chart(chart, args.output)


if __name__ == "__main__":
    main()
