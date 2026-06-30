# Tabular Foundation Benchmark

## Status

Implementation scaffold is in place. No real AutoGluon, TabICL, or TabPFN
benchmark has been run yet in this repo state.

Current incumbent from `uv run python scripts/compare.py`:

| Run | OOF | Mean CV | Features | Base models | Folds |
|---|---:|---:|---:|---:|---:|
| `20260625_231646_v010_all` | 0.9642 | 0.9562 | 39 | 9 | 5 |
| `20260626_100637_v010_all` | 0.9642 | 0.9562 | 39 | 9 | 5 |

Data audit:

| File | Status |
|---|---|
| `data/train.csv` | present |
| `data/test.csv` | present |
| `data/original.csv` | present |

Hardware audit:

| Probe | Result |
|---|---|
| `nvidia-smi` | blocked by OS (`GPU access blocked by the operating system`) |

## Implemented

| Area | Files |
|---|---|
| Standard probability artifacts | `src/stellar/blending.py` |
| Feature/backend wrappers | `src/stellar/foundation.py` |
| Benchmark runner | `scripts/benchmark_tfm.py` |
| Artifact scorer | `scripts/score_tfm.py` |
| Artifact blender | `scripts/blend_predictions.py` |
| Unified comparison | `scripts/compare.py` |
| Best-submission selector | `scripts/select_best_submission.py` |
| Benchmark configs | `config/tfm/*.yaml` |
| Tests | `tests/test_foundation.py` |
| README commands | `README.md` |

Artifacts are written under `outputs/tfm/<timestamp>_<run_name>/`:

```
config.yaml
metrics.json
classes.json
oof_proba.npy
test_proba.npy
train_ids.npy
test_ids.npy
y_true.npy
submission.csv
```

## Runbook

```bash
# Check the existing tree-stack incumbent
uv run python scripts/compare.py

# Smoke-test the new benchmark path without optional TFM dependencies
uv run python scripts/benchmark_tfm.py --config config/tfm/dummy.yaml --run-name smoke

# Rank normal runs plus TFM/blend artifacts
uv run python scripts/compare.py

# Blend two compatible probability artifacts
uv run python scripts/blend_predictions.py \
  --runs outputs/tfm/<best_run_a> outputs/tfm/<best_run_b> \
  --run-name blend_best \
  --per-class

# Submit the highest local OOF artifact
make submit-best
```

## Real Backend Commands

Install real backends only when ready to run them:

```bash
uv sync --no-dev --group tfm
```

Then start with the cheapest real runs:

```bash
uv run python scripts/benchmark_tfm.py \
  --config config/tfm/autogluon_fast.yaml \
  --run-name ag_fast

uv run python scripts/benchmark_tfm.py \
  --config config/tfm/tabicl_v2_ctx10k_raw.yaml \
  --run-name tabicl_ctx10k_raw

uv run python scripts/benchmark_tfm.py \
  --config config/tfm/tabpfn26_ctx10k_raw.yaml \
  --run-name tabpfn26_ctx10k_raw
```

## Benchmark Configs

| Config | Backend | Purpose |
|---|---|---|
| `config/tfm/dummy.yaml` | sklearn logistic regression | Offline smoke test |
| `config/tfm/autogluon_fast.yaml` | AutoGluon | Medium-quality 20 minute run |
| `config/tfm/autogluon_best.yaml` | AutoGluon | Best-quality 2 hour run |
| `config/tfm/tabicl_v2_ctx10k_raw.yaml` | TabICLv2 | 10k-row context feasibility run |
| `config/tfm/tabicl_v2_ctx50k_domain.yaml` | TabICLv2 | 50k-row domain-feature run |
| `config/tfm/tabpfn26_ctx10k_raw.yaml` | TabPFN 2.6 | 10k-row context feasibility run |

## Promotion Criteria

Promote a TFM or blend only if it satisfies all of these:

1. Writes a complete artifact directory with OOF probabilities and
   `submission.csv`.
2. Beats the incumbent OOF score by at least 0.0010, or provides a clear
   complementary probability source for a blend that does.
3. Passes `make lint && make format && make test`.
4. Has a submission message that names components and OOF score.

## Open Items

- Real TFM dependencies live in the uv `tfm` dependency group. Keep them out of
  the default environment unless you are launching TFM benchmarks.
- GPU access was blocked during the initial audit. TabICL and TabPFN may be too
  slow or infeasible on CPU for full-scale runs.
- Current blend tooling accepts standard `outputs/tfm/` artifacts. Normal
  `outputs/runs/` experiments appear in `compare.py` and `make submit-best`,
  but are not yet directly blendable without exporting their probabilities.
