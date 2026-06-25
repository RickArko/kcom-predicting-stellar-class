# Interaction Features — Experiment Report

## Recommendation

**Do not adopt interaction features as a standalone improvement.** The stacked
OOF score dropped from 0.9526 (v001) to 0.9524 (v006), despite the mean per-fold
CV score improving to 0.9515 (the best of all runs). The base models extract
more signal from the interactions, but the LogisticRegression meta-model fails
to translate that into a better stacked prediction — the same meta-model
bottleneck seen in v002 (tuned HPs). Keep the `interaction_pairs` capability
for future experiments but do not expect a stacked-OOF gain from interactions
alone.

## Results

Experiments use 3-fold stratified CV, 250 estimators per base model, and the
same hyperparameters as `config/baseline.yaml` unless noted.

| Run | Strategy | OOF | Mean CV | Feats | Time | Δ vs benchmark |
|---|---|---|---|---|---|---|
| **v001** | OHE categoricals (benchmark) | **0.9526** | 0.9509 | 18 | 132s | — |
| v006 | OHE + 7 interaction features | 0.9524 | **0.9515** | 25 | 174s | -0.0002 |
| v002 | OHE + tuned HPs | 0.9525 | 0.9527 | 18 | 316s | -0.0001 |
| baseline | Drop categoricals | 0.9521 | 0.9509 | 12 | 131s | -0.0005 |

## Experiment Details

### v006 — Interaction features

**Config:** `config/experiments/v006_interactions.yaml`

Added 7 interaction features to the v001 OHE config (same 3-fold, 250
estimators, same HPs). Two families of interactions:

| Family | Pairs | Rationale |
|---|---|---|
| redshift × colour | `redshift × u_g`, `redshift × g_r`, `redshift × r_i`, `redshift × i_z` | QSOs separate from stars/galaxies in the redshift–colour plane |
| colour × colour | `u_g × g_r`, `g_r × r_i`, `r_i × i_z` | Colour–colour diagrams are a classic SDSS discriminator |

**Code change:** Added `interaction_pairs` parameter to `ColorFeatureEngineer`
(`src/stellar/features.py`). Pairs are resolved in `fit()` — a pair is valid if
both names survive `drop_cols` or are produced by `color_pairs` (e.g. `u_g`).
Invalid pairs are dropped with a warning. Features are created in `transform()`
after colour indices and before encoding: `X[f"{a}_x_{b}"] = X[a] * X[b]`.

```
run:    v006_interactions
oof:    0.9524  (-0.0002 vs benchmark)
cv:     0.9515  (+0.0006 vs benchmark — best of all runs)
feats:  25      (+7 interaction columns)
time:   174s    (+43s vs v001)
```

**Takeaway:** The per-fold CV score improved meaningfully (0.9509 → 0.9515),
confirming the interactions add real signal to the base models. However, the
stacked OOF score went down, indicating the LogisticRegression meta-model is
the bottleneck — it cannot leverage the richer base-model probabilities better
than it already does with the simpler feature set. This echoes v002, where
per-fold CV improved (0.9527) but stacked OOF did not.

## Key Takeaways

1. **Interactions help base models but not the stack.** Mean CV improved by
   +0.0006 (best of all runs), but stacked OOF dropped by -0.0002. The
   meta-model is the limiting factor, not feature engineering.

2. **Meta-model bottleneck confirmed.** Two experiments now show per-fold
   improvements that fail to propagate through the LogisticRegression stack
   (v002 tuned HPs, v006 interactions). A meta-model variant (simple average,
   shallow tree blender) may unlock the base-model gains.

3. **Keep the capability, not the features.** The `interaction_pairs` parameter
   is now part of `ColorFeatureEngineer` and costs nothing to carry forward.
   Fold interactions into v007 (original-data augmentation) since they improve
   base-model signal — but fix the stacked-OOF ceiling by addressing the
   meta-model first.

## Implementation Notes

The `interaction_pairs` parameter added to `ColorFeatureEngineer`:

| Config key | Behavior |
|---|---|
| `features.interaction_pairs` (list of `[a, b]` pairs) | Creates `"{a}_x_{b}"` = `X[a] * X[b]` for each valid pair |
| Omitted / empty | No interaction features (backward-compatible default) |

A pair is valid if both column names survive `drop_cols` or are produced by
`color_pairs`. Invalid pairs (typos, dropped columns) are skipped with a
`warnings.warn`. The feature is created after colour indices and before
categorical encoding, so interactions can reference both raw columns
(`redshift`) and derived colour indices (`u_g`).
