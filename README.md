# kcom-predicting-stellar-class

Classify SDSS astronomical objects as **GALAXY**, **STAR**, or **QSO**.
[Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) · Metric: **balanced accuracy** · Deadline: June 30, 2026

## Results

![Submission Scores](docs/figures/submission_scores.png)

| Submission | CV (OOF) | Public LB | Private LB |
|---|---|---|---|
| benchmark (v001) | 0.9526 | 0.9544 | — |
| **final** (augment + interactions + 5-fold/1000 + threshold tuning) | **0.9641** | **0.9640** | — |

**Leaderboard context** (2,398 teams on public LB):

| Percentile | Score | My rank |
|---|---|---|
| 10th | 0.9448 | — |
| 25th | 0.9548 | benchmark ~25th pct |
| **50th (median)** | **0.9637** | **final ~49th pct (rank ~1,181)** |
| 75th | 0.9676 | — |
| 90th | 0.9714 | — |

Private LB populates after the competition ends. Regenerate after new submissions:

```bash
make visualize
```

**Key levers** (benchmark → final, +0.0115 OOF):
- **Per-class threshold tuning** (+0.0070) — Nelder-Mead simplex search on OOF probabilities, directly optimizing balanced accuracy instead of plain argmax
- **5-fold / 1000 estimators** (+0.0039) — better base models and OOF estimates
- **Original-data augmentation** (+0.0004) — 100k real SDSS17 rows appended to 577k synthetic
- **Interaction features** (+0.0002) — `redshift × colour`, `colour × colour` (helps only when OHE categoricals are dropped)

See [`Iteration.md`](Iteration.md) for the full experiment log and [`docs/experiments/`](docs/experiments/) for per-topic reports.

## Submission

All five improvements combined in [`config/experiments/final.yaml`](config/experiments/final.yaml), targeting **0.9714** (top 10%):

| Technique | Where | Config key | Expected gain |
|---|---|---|---|
| **Seed averaging** | `train.py:_expand_seeds` | `lgbm.seeds: [42,43,44]` | +0.002–0.005 |
| **Target encoding** | `features.py:137-153` | `encoding: target` | +0.001–0.003 |
| **Feature transforms** | `features.py:52-64` | `ratio_pairs`, `log_transform_cols`, `poly_cols`/`polynomial_degree` | +0.001–0.003 |
| **Calibrated meta** | `train.py:99-103` | `meta.calibrated: true` | +0.001–0.002 |
| **Pseudo-labeling** | `train.py:_run_pseudo_labeling` | `pseudo_label: {enabled: true, confidence_threshold: 0.95}` | +0.005–0.015 |

```bash
make train CONFIG=config/experiments/final.yaml RUN_NAME=v010_all
```

**Then to submit the experiment for `leaderboard-score`:**

```bash
make submit SUBMISSION_FILE=outputs/runs/20260626_100637_v010_all/submission.csv \
           SUBMISSION_MSG="v010_all: seeds+target_enc+poly+pseudo_label+calibrated"
```

## Quick Start

```bash
make install          # uv sync + kaggle auth (one-time)
make download         # fetch competition data (one-time)
make train            # train ensemble → submission.csv
make submit           # upload to Kaggle + show leaderboard
```

**Happy path** (full pipeline in one command): `make all`

## Beat the Benchmark

```bash
# 1. Download the original SDSS17 dataset for augmentation
uv run kaggle datasets download -d fedesoriano/stellar-classification-dataset-sdss17 -p data/
unzip -o data/stellar-classification-dataset-sdss17.zip -d data/ && mv data/star_classification.csv data/original.csv

# 2. Train the final model (~22 min on CPU)
make train CONFIG=config/experiments/final.yaml RUN_NAME=final

# 3. Compare against all prior experiments
uv run python scripts/compare.py

# 4. Submit
make submit SUBMISSION_FILE=outputs/runs/<timestamp>_final/submission.csv \
           SUBMISSION_MSG="final: augment + interactions + 5-fold/1000 + threshold tuning"
```

Re-predict without retraining:

```bash
uv run python scripts/predict.py --run-dir outputs/runs/<timestamp>_final
```

## Tabular Foundation Benchmarks

TFM and AutoML benchmark runs write probability artifacts under `outputs/tfm/`
with OOF probabilities, test probabilities, metrics, and `submission.csv`.

```bash
# 1. Check the current tree-stack incumbent
uv run python scripts/compare.py

# 2. Offline smoke test for the benchmark runner
uv run python scripts/benchmark_tfm.py --config config/tfm/dummy.yaml --run-name smoke

# 3. Install real backends only when needed
uv sync --no-dev --group tfm

# 4. Run candidate benchmarks
uv run python scripts/benchmark_tfm.py \
  --config config/tfm/autogluon_fast.yaml \
  --run-name ag_fast

uv run python scripts/benchmark_tfm.py \
  --config config/tfm/tabicl_v2_ctx10k_raw.yaml \
  --run-name tabicl_ctx10k_raw

uv run python scripts/benchmark_tfm.py \
  --config config/tfm/tabpfn26_ctx10k_raw.yaml \
  --run-name tabpfn26_ctx10k_raw

# 5. Score and rank normal runs plus TFM/blend artifacts
uv run python scripts/compare.py

# 6. Blend the best probability artifacts, then re-score
uv run python scripts/blend_predictions.py \
  --runs outputs/tfm/<best_run_a> outputs/tfm/<best_run_b> \
  --run-name blend_best \
  --per-class

uv run python scripts/compare.py
make visualize

# 7. Submit the winning artifact
make submit-best
```

Real AutoGluon, TabICL, and TabPFN runs can be slow and may require GPU access,
model-license acceptance, or package-specific credentials. Keep the dummy run as
the first sanity check before launching paid-in-runtime experiments.

## Kaggle API Setup

```bash
# Option A: environment variable
export KAGGLE_API_TOKEN=KGAT_<your-token>

# Option B: token file
echo -n "KGAT_<your-token>" > .kaggle/access_token
chmod 600 .kaggle/access_token

# Get your token at https://www.kaggle.com/settings → API → Create New Token
```

You must **join the competition** (Accept Rules) on the Kaggle page before `make download` works.

## Pipeline

```mermaid
flowchart TD
    A[Raw Data] --> B[Feature Engineering]
    A --> AUG[SDSS17 Augmentation]

    subgraph FE[Feature Engineering]
        direction LR
        B --> C[drop ID/metadata cols]
        B --> D[photometric colour indices<br>u-g, g-r, r-i, i-z]
        B --> R[band ratios<br>u/g, g/r, r/i, i/z]
        B --> L[log transforms<br>log₁₀ redshift]
        B --> P[poly features deg 2<br>u², u·g, …]
        B --> I[interaction features<br>redshift × colour]
        B --> ENC[encoding:<br>target / OHE / label]
    end

    FE --> G[Stratified 5-Fold CV]

    subgraph SE[Seed Bagging — 9 Base Models]
        direction LR
        S1[LGBM s0] --- S2[LGBM s1] --- S3[LGBM s2]
        S4[XGB s0] --- S5[XGB s1] --- S6[XGB s2]
        S7[CB s0] --- S8[CB s1] --- S9[CB s2]
    end

    G --> SE

    SE --> OOF[OOF Probabilities<br>per fold per model]
    OOF --> META[Meta-Model<br>SimpleAverage / LogisticRegression<br>+ optional CalibratedClassifierCV]
    OOF --> THRESH[Per-Class<br>Threshold Tuning<br>Nelder-Mead]
    META --> FINAL[Final Predictions]
    THRESH --> FINAL

    FINAL --> PL{Pseudo-Label<br>confidence ≥ 0.95?}
    PL -->|yes| AUG
    PL -->|no| SUB[submission.csv]

    AUG --> A
```

- **Features** — drop metadata/ID, derive SDSS colour indices (`u-g, g-r, r-i, i-z`), add `redshift × colour` interactions
- **Base models** — LightGBM, XGBoost, CatBoost (5-fold stratified CV, 1000 estimators)
- **Meta** — simple average of base-model probabilities + per-class threshold tuning
- **Augmentation** — 100k original SDSS17 rows concatenated with 577k synthetic training rows

## Development

```bash
make lint           # ruff check
make format         # ruff format --check
make format-fix     # apply formatting
make test           # pytest (synthetic data, no Kaggle needed)
make visualize      # regenerate submission score chart
```

Submit a custom file:

```bash
make submit SUBMISSION_FILE=outputs/runs/<name>/submission.csv SUBMISSION_MSG="description"
```

Submit the highest local OOF `submission.csv` from either `outputs/runs/` or
`outputs/tfm/`:

```bash
make submit-best
```

## Repository Structure

```
config/
  config.yaml              # tuned default (5-fold, 1000 estimators)
  baseline.yaml            # fast reference (3-fold, 250 estimators)
  experiments/             # v001–v009 + final.yaml
  tfm/                     # tabular foundation / AutoML benchmark configs
src/stellar/               # data.py, features.py, models.py, tracking.py
scripts/                   # train.py, predict.py, compare.py, benchmark_tfm.py, score_tfm.py
tests/                     # unit + integration (synthetic SDSS-like data)
docs/experiments/          # per-topic experiment reports
outputs/runs/              # timestamped run artifacts (config, metrics, model, submission)
outputs/tfm/               # TFM/blend artifacts with OOF/test probabilities
```

## Submission

Here's a one-liner per improvement:
Technique	Where	Config key
Seed averaging	train.py:_expand_seeds	lgbm.seeds: [42,43,44]
Target encoding	features.py:137-153	encoding: target
Feature transforms	features.py:52-64	ratio_pairs, log_transform_cols, poly_cols/polynomial_degree
Calibrated meta	train.py:99-103	meta.calibrated: true
Pseudo-labeling	train.py:_run_pseudo_labeling	pseudo_label: {enabled: true, confidence_threshold: 0.95}
Combined in config/experiments/final.yaml targeting 0.9714 (top 10%). Run with:
