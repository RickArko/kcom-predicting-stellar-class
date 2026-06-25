# Iteration Workflow

A structured workflow for running, tracking, and comparing model experiments for the
[Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6) competition.

---

## Quick Start

```bash
make install          # dependencies + kaggle auth (one-time)
make download         # fetch competition data (one-time)

# Run your first experiment
make train CONFIG=config/baseline.yaml RUN_NAME=my_first_experiment

# Compare all experiments
uv run python scripts/compare.py
```

---

## How It Works

### 1. Config-driven experiments

Every experiment is defined by a single YAML file.  The config controls every
knob: which features to use, how many CV folds, every model hyperparameter.

```bash
make train CONFIG=config/experiments/v001_keep_categoricals.yaml RUN_NAME=v001
```

### 2. Automatic run tracking

Each `make train` creates a timestamped directory under `outputs/runs/`:

```
outputs/runs/
  20260624_131300_baseline/
    config.yaml          # frozen copy of the config used
    metrics.json         # OOF scores, params, wall time
    models/ensemble.joblib  # serialised ensemble (loadable)
    submission.csv       # competition submission
```

This makes every experiment **reproducible** — you can re-run from the saved
config, or re-predict from the saved model.

### 3. Compare experiments

```bash
uv run python scripts/compare.py

# Example output:
#                                  run  overall_oof_score  ...  n_features
#    20260624_131519_v001_keep_categoricals            0.9526  ...          18
#    20260624_131920_v002_tuned_hyperparams            0.9525  ...          18
#                 20260624_131300_baseline            0.9521  ...          12
```

### 4. Re-predict from a saved model (no re-training)

```bash
uv run python scripts/predict.py --run-dir outputs/runs/20260624_131300_baseline
```

---

## Running the Baseline

The baseline is the "original" pipeline before any iteration:

```bash
make train CONFIG=config/baseline.yaml RUN_NAME=baseline
```

| Setting | Value |
|---|---|
| Features | 5 photometric bands + 4 SDSS colour indices + alpha + delta + redshift |
| Categoricals | Dropped (`spectral_type`, `galaxy_population`) |
| Models | LGBM + XGBoost + CatBoost (stacked with LogisticRegression) |
| CV | 3-fold stratified (for speed; use 5 for final) |
| Estimators per model | 250 |

Expected OOF balanced accuracy: **~0.952**

---

## Iteration Log

### v001 — Keep categorical features (`spectral_type`, `galaxy_population`)

**Hypothesis:** `spectral_type` (M, O/B, G/K, A/F) and `galaxy_population`
(Red_Sequence, Blue_Cloud) carry strong class signal that the original pipeline
was discarding.

**Change:** Removed `spectral_type` and `galaxy_population` from `drop_cols`,
added `cat_cols` so they get one-hot encoded.  Config at
`config/experiments/v001_keep_categoricals.yaml`.

```yaml
# config/experiments/v001_keep_categoricals.yaml
features:
  drop_cols:
    - id
    - obj_ID
    # ... (spectral_type and galaxy_population removed)
  cat_cols:
    - spectral_type
    - galaxy_population
```

```
run:    v001_keep_categoricals
oof:    0.9526  (+0.0005 vs baseline)
feats:  18      (+6 one-hot columns)
time:   132s    (same; small feature set)
```

**Takeaway:** Small but real improvement.  The tree models already capture
non-linear relationships from photometry, so the categoricals add limited
orthogonal signal.  Worth keeping, but don't expect a jump.

---

### v002 — Tune hyperparameters for better generalisation

**Hypothesis:** Lower learning rate, more estimators, stronger regularisation,
and deeper trees should improve OOF scores.

**Change:** Config at `config/experiments/v002_tuned_hyperparams.yaml`.

| Param | Baseline | v002 |
|---|---|---|
| `n_estimators` | 250 | 500 |
| `learning_rate` | 0.05 | 0.03 |
| `max_depth` | 6 | 8 |
| `subsample` | 0.8 | 0.7 |
| `reg_alpha/lambda` | 0.1 | 0.5 |
| `meta.C` | 1.0 | 0.5 |
| `meta.max_iter` | 1000 | 2000 |

```
run:    v002_tuned_hyperparams
oof:    0.9525  (same as v001)
feats:  18
time:   316s    (2.4× slower)
```

**Takeaway:** Per-fold CV scores improved (~0.9509 → ~0.9527) but the stacked
OOF score did not.  This indicates the meta-model may be overfitting the OOF
probabilities with the extra regularisation.  The simple average of base models
might match or beat the logistic regression stack.

---

### v003 — Label encoding (ordinal)

**Hypothesis:** Ordinal label encoding is simpler than OHE and preserves
feature count (14 vs 18), which may reduce meta-model overfitting.

**Change:** Set `encoding: label` in config.  Replaces `spectral_type` and
`galaxy_population` with integer codes (0-3, 0-1) instead of 6 OHE columns.
Config at `config/experiments/v003_label_encoding.yaml`.

Code change: Added `encoding` parameter (`"ohe"`, `"label"`, `"passthrough"`)
to `ColorFeatureEngineer` in `src/stellar/features.py`.

```
run:    v003_label_encoding
oof:    0.9524  (+0.0003 vs baseline, -0.0002 vs OHE)
feats:  14
time:   145s
```

**Takeaway:** Ordinal encoding is slightly worse than OHE (0.9524 vs 0.9526).
The false ordering imposed on categories likely forces trees to split harder
to compensate.  OHE is marginally better.

---

### v004 — CatBoost native categoricals

**Hypothesis:** CatBoost's native handling of categorical features (via
`cat_features`) should outperform generic OHE because CatBoost uses
ordered-target statistics for splitting.

**Change:** Same features as v003, but pass `cat_features: [8, 9]` to
CatBoost's `fit()` as a `model_fit_kwarg`.  LGBM/XGBoost see the columns
as numeric ordinals.  Config at `config/experiments/v004_cb_native.yaml`.

Code change: `StackingEnsemble.fit()` accepts `model_fit_kwargs` to support
per-model `fit()` params (needed because CatBoost's `cat_features` can't
survive `sklearn.base.clone`).

```
run:    v004_cb_native
oof:    0.9525  (+0.0004 vs baseline)
feats:  14
time:   210s  (CatBoost native training is slower)
```

**Takeaway:** Native CatBoost handling scores 0.9525 — same as OHE within
noise.  For 4+2-value categories the native handling provides no measurable
benefit over OHE.

---

### v005 — LightGBM native categoricals (invalid)

**Hypothesis:** LightGBM's `categorical_feature` param should improve splits
over ordinal encoding.

**Change:** Same features as v003, add `categorical_feature: [8, 9]` to LGBM
config.  Config at `config/experiments/v005_lgb_native.yaml`.

```
run:    v005_lgb_native
oof:    0.9524  (same as v003 — param was ignored)
feats:  14
time:   135s
```

**Result:** LightGBM ≥4.6 ignores `categorical_feature` in the sklearn wrapper
constructor (warning: "will be ignored").  The run is a duplicate of v003.
Not a valid test of native categoricals.

To properly test LGBM native categoricals, convert columns to
`pd.Categorical` dtype (e.g. a new `encoding: category` mode in
`ColorFeatureEngineer`) and let LGBM auto-detect.

---

### Summary

| Run | OOF | Features | Time | Delta |
|---|---|---|---|---|
| baseline | 0.9521 | 12 | 131s | — |
| v001 (OHE) | 0.9526 | 18 | 132s | +0.0005 |
| v002 (tuned HP) | 0.9525 | 18 | 316s | +0.0004 |
| v003 (label encode) | 0.9524 | 14 | 145s | +0.0003 |
| v004 (CB native) | 0.9525 | 14 | 210s | +0.0004 |
| v005 (LGB native)* | 0.9524 | 14 | 135s | +0.0003 |

*Param ignored by LGBM ≥4.6 — effectively v003.

All strategies that keep categoricals cluster at 0.9524–0.9526.  **OHE remains
the simplest and marginally best choice.**  Diminishing returns are severe at
this level — expect single basis points at most from further categorical
tuning.  See `docs/experiments/categoricals.md` for the full report.

---

## Workflow Reference

### Run an experiment

```bash
make train CONFIG=config/experiments/my_config.yaml RUN_NAME=my_experiment
```

### Compare results

```bash
uv run python scripts/compare.py
uv run python scripts/compare.py --sort-by elapsed_seconds
```

### Re-predict from a saved model

```bash
uv run python scripts/predict.py --run-dir outputs/runs/20260624_131300_baseline
```

### Submit to Kaggle

```bash
make submit
make submit SUBMISSION_FILE=outputs/runs/20260624_131300_baseline/submission.csv \
             SUBMISSION_MSG="baseline: photometric bands + color indices"
```

### Create a new experiment config

1. Copy an existing config: `cp config/baseline.yaml config/experiments/my_idea.yaml`
2. Edit the feature / model / CV sections
3. Run it: `make train CONFIG=config/experiments/my_idea.yaml`

---

## Next Directions

### Likely to help

- **CatBoost with native categoricals** — currently one-hot encoding is applied
  uniformly; CatBoost handles raw categoricals natively, and LGBM has a
  `categorical_feature` parameter.  A per-model `cat_cols` pass-through could
  improve over generic OHE.
- **Interaction features** — `redshift × colour_index` or
  `redshift × spectral_type` could surface class-separating structure that
  individual features miss.
- **Target encoding** — encode `spectral_type` by its per-class probability
  (with CV to avoid leakage) instead of one-hot.
- **Feature selection** — LGBM/XGB feature importance can identify which of
  the 6 OHE columns carry weight; drop the low-signal ones.

### Worth trying

- **LightGBM native categorical support** — pass `categorical_feature` in the
  config instead of OHE.
- **5-fold CV** — 3-fold was used for iteration speed; final runs should use
  5-fold for better OOF estimates and test predictions.
- **Stacking meta-model variants** — try a simple neural net or a shallow tree
  as the blender instead of logistic regression.
- **Post-processing** — calibrate probabilities with Platt scaling or isotonic
  regression before the final argmax.

### Architecture notes

- `ColorFeatureEngineer` is a sklearn `Transformer`, so it composes naturally
  in a `Pipeline`.  Future experiments can add `SelectKBest`, `PCA`, or
  `PolynomialFeatures` as additional pipeline steps without touching model code.
- `StackingEnsemble` serialises with joblib.  Model artifacts are versioned
  alongside their config and metrics in each run directory.
- The `scripts/compare.py` utility reads from `outputs/runs/` — no database or
  external tracking system needed for basic iteration.
