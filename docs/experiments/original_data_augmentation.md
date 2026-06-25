# Original-Data Augmentation — Experiment Report

## Recommendation

**Adopt original-data augmentation.** Augmenting the 577k synthetic training
rows with 100k rows from the original SDSS17 dataset produced the best OOF
score of all experiments (0.9532). The gain is modest (+0.0004 over the
no-augment control, +0.0006 over the v001 benchmark) rather than the
+0.01–0.03 hoped in the plan — the synthetic data already subsumes most of the
original's signal. Still, it is the current best and costs nothing at inference
time. Use `config/experiments/v007_augment.yaml` as the new baseline for future
experiments.

## Results

All runs use 3-fold stratified CV, 250 estimators per base model. v007 runs
drop categoricals (schema gap) and include 7 interaction features.

| Run | Augment | OOF | Mean CV | Feats | Train rows | Time | Δ vs v001 |
|---|---|---|---|---|---|---|---|
| **v007_augment** | yes | **0.9532** | 0.9514 | 19 | 677,347 | 156s | **+0.0006** |
| v007_no_augment | no | 0.9528 | 0.9515 | 19 | 577,347 | 157s | +0.0002 |
| v001 (benchmark) | — | 0.9526 | 0.9509 | 18 | 577,347 | 132s | — |
| v006 (cats + interactions) | — | 0.9524 | 0.9515 | 25 | 577,347 | 174s | -0.0002 |
| baseline (drop cats) | — | 0.9521 | 0.9509 | 12 | 577,347 | 131s | -0.0005 |

**Augmentation effect (single-axis comparison):** v007_no_augment → v007_augment
= +0.0004 OOF, with 0 rows deduped (no exact duplicates between synthetic and
original).

## Experiment Details

### Original dataset

- **Source:** [`fedesoriano/stellar-classification-dataset-sdss17`](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
  ("Stellar Classification Dataset - SDSS17", 100k rows, 7.2MB)
- **Competition attribution:** The competition page states the data "was
  inspired by the Stellar classification dataset."
- **Schema:** `obj_ID, alpha, delta, u, g, r, i, z, run_ID, rerun_ID, cam_col,
  field_ID, spec_obj_ID, class, redshift, plate, MJD, fiber_ID`

### Schema gap

| Column | Synthetic | Original | Handling |
|---|---|---|---|
| `alpha, delta, u, g, r, i, z, redshift` | yes | yes | shared features |
| `class` | yes | yes | target |
| `obj_ID, run_ID, ...` (metadata) | yes | yes | in `drop_cols` (ignored) |
| `id` | yes | no | in `drop_cols` (ignored) |
| `spectral_type` | yes | **no** | **dropped** for v007 |
| `galaxy_population` | yes | **no** | **dropped** for v007 |

`spectral_type` and `galaxy_population` are synthetic-derived columns not
present in the original SDSS data. Dropping them gives a clean single-axis
test (augmentation on/off) and is consistent with the finding from
`docs/experiments/categoricals.md` that categoricals add only +0.0005.

### v007_no_augment — Control (no augmentation)

**Config:** `config/experiments/v007_no_augment.yaml`

Drops categoricals, adds 7 interaction features, no augmentation. This is the
apples-to-apples control for the augmentation effect.

```
run:    v007_no_augment
oof:    0.9528  (+0.0002 vs v001 benchmark)
feats:  19
time:   157s
```

**Surprise:** This run beat the v001 benchmark (0.9526) despite dropping
categoricals. The interaction features (which hurt in v006 when combined with
OHE categoricals) actually help when categoricals are dropped: 0.9521 (baseline,
no cats, no interactions) → 0.9528 (no cats, + interactions) = +0.0007. The OHE
categorical columns appear to interfere with the interaction features in the
meta-model.

### v007_augment — With original-data augmentation

**Config:** `config/experiments/v007_augment.yaml`

Same as v007_no_augment but with `data.augment_path: data/original.csv` and
`data.dedup_cols: [alpha, delta, u, g, r, i, z, redshift]`.

```
run:    v007_augment
oof:    0.9532  (+0.0004 vs control, +0.0006 vs v001 benchmark)
feats:  19
train:  677,347  (577,347 synthetic + 100,000 original, 0 deduped)
time:   156s
```

**Code change:** Extended `load_data()` in `src/stellar/data.py` with
`augment_path` and `dedup_cols` parameters. When `augment_path` is set, the
original CSV is concatenated with the synthetic train, then deduplicated on
`dedup_cols` with `keep="first"` (synthetic rows retained for exact duplicates).
`scripts/train.py` reads the new `data` config section and passes the params
through. `scripts/predict.py` is unaffected (test data is never augmented).

## Key Takeaways

1. **Augmentation helps but less than expected.** The +0.0004 gain is far below
   the +0.01–0.03 typical for Playground competitions. The synthetic data (577k
   rows) already captures most of the original's (100k rows) signal — the 17%
   increase in training data adds only marginal orthogonal information.

2. **Interactions + no categoricals > OHE categoricals.** The unexpected
   finding: dropping categoricals and adding interactions (0.9528) beats OHE
   categoricals alone (0.9526). In v006, interactions + OHE categoricals hurt
   (0.9524). The OHE columns likely interfere with the interaction features in
   the LogisticRegression meta-model — dropping them lets the interaction
   signal propagate cleanly.

3. **0 deduped rows.** No exact duplicates between synthetic and original,
   confirming the synthetic data is generated (not copied). The dedup
   mechanism is still valuable as a safety net for future datasets.

4. **New best config:** `config/experiments/v007_augment.yaml` at OOF 0.9532.
   Use this as the baseline for future experiments (5-fold final runs, threshold
   tuning, meta-model variants).

## Implementation Notes

### Config knobs

```yaml
data:
  augment_path: data/original.csv   # path to original dataset (omit for no augment)
  dedup_cols: [alpha, delta, u, g, r, i, z, redshift]  # dedup on these columns
```

### Data loader changes

`load_data()` in `src/stellar/data.py` accepts two new optional parameters:

| Parameter | Default | Behavior |
|---|---|---|
| `augment_path` | `None` | Path to a CSV concatenated with the synthetic train |
| `dedup_cols` | `None` | Columns for `drop_duplicates(keep="first")` after concat |

When `augment_path` is `None` (default), `load_data` is backward-compatible —
existing configs without a `data` section work unchanged.

### Downloading the original dataset

```bash
uv run kaggle datasets download -d fedesoriano/stellar-classification-dataset-sdss17 -p data/
cd data && unzip stellar-classification-dataset-sdss17.zip && mv star_classification.csv original.csv
```

`data/original.csv` is gitignored (covered by `data/*.csv` in `.gitignore`).
