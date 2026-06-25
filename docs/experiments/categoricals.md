# Categorical Feature Processing — Experiment Report

## Recommendation

**Use one-hot encoding** (`cat_cols` with `encoding: ohe`, i.e. the current
v001 config).  It is the simplest strategy, adds only ~0.0005 over dropping
categoricals, and more complex strategies (label encoding, CatBoost native,
LightGBM native) all produce the same OOF score within noise.  No strategy
clearly beats OHE.

## Results

Experiments use 3-fold stratified CV, 250 estimators per base model, and the
same hyperparameters as `config/baseline.yaml` unless noted.

| Run | Strategy | OOF | Mean CV | Feats | Time | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | Drop categoricals | 0.9521 | 0.9509 | 12 | 131s | — |
| v001 | One-hot encode | 0.9526 | 0.9509 | 18 | 132s | +0.0005 |
| v002 | OHE + tuned HPs | 0.9525 | 0.9527 | 18 | 316s | +0.0004 |
| v003 | Label encode (ordinal) | 0.9524 | 0.9509 | 14 | 145s | +0.0003 |
| v004 | Label encode + CatBoost `cat_features` | 0.9525 | 0.9510 | 14 | 210s | +0.0004 |
| v005* | Label encode + LGBM `categorical_feature` | 0.9524 | 0.9509 | 14 | 135s | +0.0003 |

*`categorical_feature` param is ignored by LightGBM ≥4.6 in the sklearn
wrapper.  Run v005 is effectively a duplicate of v003.

## Experiment Details

### v003 — Label encoding

**Config:** `config/experiments/v003_label_encoding.yaml`

Replace `spectral_type` (4 categories) and `galaxy_population` (2 categories)
with ordinal integers via `LabelEncoder`.  All three tree models receive the
same numeric columns — they treat them as ordinal features.

**Code change:** Added `encoding` parameter to `ColorFeatureEngineer`
(`"ohe"`, `"label"`, `"passthrough"`).

**Result:** 0.9524 — slightly below OHE (0.9526).  Ordinal encoding imposes a
false ordering on categories that trees must work around.

### v004 — CatBoost native categoricals

**Config:** `config/experiments/v004_cb_native.yaml`

Label-encode cat_cols to integers (same as v003), but pass `cat_features: [8, 9]`
to CatBoost via `model_fit_kwargs` so CatBoost treats them as categorical
(not ordinal).  LGBM and XGBoost see them as numeric ordinals.

**Code change:** `StackingEnsemble.fit()` accepts `model_fit_kwargs` dict to
support per-model `fit()` params that cannot survive `sklearn.base.clone`
(CatBoost's `cat_features` notably).

**Result:** 0.9525 — same as OHE within noise.  CatBoost's native handling
does not meaningfully improve on OHE for these 4+2-value categories.

### v005 — LightGBM native categoricals

**Config:** `config/experiments/v005_lgb_native.yaml`

Label-encode cat_cols, pass `categorical_feature: [8, 9]` to LGBM.

**Result:** LightGBM ≥4.6 ignores `categorical_feature` in the sklearn wrapper
constructor params (it must be passed to the `Dataset` constructor instead).
The run is effectively identical to v003 (0.9524).  Not a valid native-categorical
test.

## Key Takeaways

1. **Categoricals add marginal value.** The jump from dropping (0.9521) to OHE
   (0.9526) is 0.0005 at best.  Tree models already extract class-separating
   signal from photometric bands + redshift + colour indices.

2. **Native categorical support doesn't matter here.**  With only 6 category
   values total (4+2), there is no high-cardinality problem.  OHE, label
   encoding, and native CatBoost all converge to 0.9524–0.9526.

3. **Keep it simple.**  OHE is the default (`encoding: ohe`) and requires no
   special model params.  Stick with it.

4. **Skipped strategies:**
   - **Target encoding** — high implementation complexity for expected gain
     of ≤0.0001.  Not worth it.
   - **LightGBM native** — neutralised by API deprecation.  If revisited,
     use `pd.Categorical` dtype and let LightGBM auto-detect.

## Implementation Notes

The `encoding` parameter added to `ColorFeatureEngineer`:

| Mode | Behavior |
|---|---|
| `"ohe"` (default) | One-hot via `OneHotEncoder` |
| `"label"` | Ordinal via `LabelEncoder` → `int32` |
| `"passthrough"` | Keep raw strings unchanged |

To use CatBoost native categoricals in a future experiment, add `cat_features`
to the catboost section of the config — `scripts/train.py` automatically
extracts it and passes it as a `model_fit_kwarg` to `StackingEnsemble.fit()`.
