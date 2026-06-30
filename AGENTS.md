# AGENTS.md — kcom-predicting-stellar-class

Kaggle Playground S6E6 — classify SDSS objects as **GALAXY / STAR / QSO**.
Metric: **balanced accuracy**. Deadline: **June 30, 2026** (imminent).

## Commands

All Python invocations **must** be prefixed with `uv run` (uv manages the env;
`.venv` exists but is not on PATH — bare `python`/`pytest` will fail or hit the
wrong interpreter).

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev` + editable install + kaggle auth check |
| `make download` | Fetch/expand competition CSVs into `data/` (requires auth + competition joined) |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train ensemble & save a run |
| `make train ARGS="--flag x"` | Pass extra args to `scripts/train.py` |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run pytest tests/ -v` |
| `make test ARGS="-k name -x"` | Focused test run |
| `make lint` | `ruff check src/ scripts/ tests/` (check only, no autofix) |
| `make format` | `ruff format ... --check` (check only — does NOT rewrite) |
| `make format-fix` | Apply ruff formatting |
| `make visualize` | Regenerate `docs/figures/submission_scores.png` from leaderboard |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle + show leaderboard |

Verification loop after changes: `make lint && make format && make test`.
There is no typechecker or mypy configured.

## Two toolchains, two envs (do not mix)

`pyproject.toml` defines a `dev` dependency-group and a `tfm` dependency-group
that are marked **mutually exclusive** under `[tool.uv] conflicts`.

- Default work (train/predict/test/lint): `make install` runs `uv sync --extra dev`.
- TFM/AutoML benchmarks only: `uv sync --no-dev --group tfm` (installs
  `autogluon.tabular`, `tabicl`, `tabpfn`). Switching groups re-resolves the env.
- TFM backends can be slow / need GPU / need license acceptance. Always run the
  offline **dummy** sanity check first:
  `uv run python scripts/benchmark_tfm.py --config config/tfm/dummy.yaml --run-name smoke`.

## Architecture

- **Package** `src/stellar/`: `data.py` (loaders + augmentation), `features.py`
  (`ColorFeatureEngineer`), `models.py` (`StackingEnsemble` + meta-models +
  `MODEL_REGISTRY`), `tracking.py` (run logger), `foundation.py` (TFM
  `BenchmarkBackend` subclasses: Dummy/AutoGluon/TabICL/TabPFN),
  `blending.py` (probability-artifact blending). Installed editable via
  hatchling (`packages = ["src/stellar"]`).
- **Ensemble** (`StackingEnsemble`): base models → meta-model on out-of-fold
  probabilities, stratified k-fold CV. Supports optional per-class threshold
  tuning (`tune_thresholds`, Nelder-Mead on OOF probs — the single biggest
  lever, +0.0070 OOF) and iterative pseudo-labeling
  (`predict_proba_base_avg` feeds confident test rows back as training data).
- **Two config formats coexist** in `scripts/train.py` `_build_ensemble`:
  - Legacy: top-level `lgbm` / `xgb` / `catboost` sections (see `config.yaml`).
  - New: a `models:` list of `{type: ...}` entries — needed for diversity
    (ExtraTrees, HistGBM, Ridge, multi-seed bags). `type` must be in
    `MODEL_REGISTRY`. Old configs still work via dispatch.
- **Meta-model** via `meta.model`: `logistic_regression` (default) |
  `simple_average` | `gradient_boosting` | `random_forest` | `weighted_average`.
  `meta.calibrated: true` wraps LogisticRegression in `CalibratedClassifierCV`.
  `meta.tune_thresholds: true` enables per-class Nelder-Mead threshold search.
- **Seed bagging**: any model section accepts `seeds: [42, 43, 44]` instead of a
  single `random_state`; `_expand_seeds` fans out one instance per seed.
- **Config-driven**: YAML in `config/` controls features, CV, and every model
  hyperparam. `config/config.yaml` = tuned default (5-fold, 1000 estimators);
  `config/baseline.yaml` = fast reference (3-fold, 250 estimators);
  `config/experiments/final.yaml` = **current best** (see Experiment state).
  `config/tfm/` holds TFM/AutoML benchmark configs.
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` with a frozen
  `config.yaml`, `metrics.json`, `models/ensemble.joblib`, and `submission.csv`.
  Also writes the canonical `outputs/submissions/submission.csv`.
- **TFM/benchmark runs** are **not** `make train` — invoke
  `scripts/benchmark_tfm.py` directly; artifacts land in `outputs/tfm/<run>/`
  (OOF probs, test probs, metrics, `submission.csv`). Score with
  `scripts/score_tfm.py --top N`, blend with `scripts/blend_predictions.py
  --runs <a> <b> --run-name x --per-class`.
- **Re-predict without retraining**: `uv run python scripts/predict.py --run-dir outputs/runs/<name>`.
- **Compare runs**: `uv run python scripts/compare.py` (reads `outputs/runs/`).
  Flags: `--sort-by elapsed_seconds`, `--feature-importance` (permutation
  importance on the best run's OOF meta-features).
- **Adversarial validation**: `uv run python scripts/adversarial_validate.py`
  (train-vs-test distribution shift diagnostic, ~2 min, no score impact).

## Conventions & gotchas

- `ColorFeatureEngineer` `encoding` param: `"ohe"` (default), `"label"`
  (LabelEncoder → int32), `"passthrough"` (raw strings), `"target"` (smoothed
  target-mean encoding — **requires `y` in `fit_transform`**; `train.py` only
  passes it when `encoding == "target"`). Set via `features.encoding` in config.
- `ColorFeatureEngineer` also builds (all optional, config-driven): `color_pairs`
  (u-g, g-r, ...), `ratio_pairs` (u/g, ...), `log_transform_cols`,
  `poly_cols` + `polynomial_degree`, `interaction_pairs` (e.g. redshift × colour).
- **Augmentation**: `data.augment_path` + `data.dedup_cols` in config append
  real SDSS17 rows from `data/original.csv` (downloaded separately via the
  `fedesoriano/stellar-classification-dataset-sdss17` Kaggle dataset — not part
  of `make download`).
- `StackingEnsemble.fit()` accepts `model_fit_kwargs` (dict keyed by model name)
  for per-model `fit()` kwargs that can't survive `sklearn.base.clone`.
  `scripts/train.py` auto-extracts `cat_features` from the `catboost` config
  section and passes it as a fit kwarg — **do not pass `cat_features` to the
  CatBoost constructor**.
- **LightGBM ≥4.6 ignores `categorical_feature`** in the sklearn wrapper
  constructor (warns "will be ignored"). Don't rely on it; use OHE/target
  encoding or convert columns to `pd.Categorical` for native handling.
- `catboost_info/` is written during training and is gitignored — safe to delete.
- `from __future__ import annotations` is used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- Runtime: Python 3.13 (`.python-version`); `requires-python = ">=3.11"`.

## Testing

- Tests use **synthetic SDSS-like data** (the `synthetic_data` fixture in
  `tests/test_integration.py`) — `make test` works offline, no Kaggle download
  needed.
- `tests/test_models.py` covers the feature transformer, ensemble, meta-models,
  and save/load roundtrip; `tests/test_integration.py` covers the end-to-end
  pipeline + submission format; `tests/test_data.py` covers loaders/augmentation;
  `tests/test_foundation.py` covers TFM backends (uses `DummyBackend`).

## Kaggle auth

- Token file: `.kaggle/access_token` (chmod 600), or env var `KAGGLE_API_TOKEN`.
  `make download` / `make submit` read the file first, then fall back to the env
  var. Legacy `~/.kaggle/kaggle.json` also works.
- You must **join the competition** (Accept Rules) on the Kaggle web page before
  `make download` works — otherwise it 403s.

## Gitignored artifacts (won't appear in `git status`)

`data/*.csv` and `*.zip`, `outputs/submissions/*` (except `.gitkeep`),
`outputs/runs/*`, `outputs/tfm/*` (artifacts), `models/`, `*.joblib`/`*.pkl`/
`*.cbm`, `catboost_info/`, `.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`,
caches. Run data lives only on disk — re-run `make download` on a fresh clone.

## Experiment state

Before new experiments, read **`Iteration.md`** (full log + takeaways) and
**`docs/experiments/`** (`categoricals.md`, `interactions.md`,
`original_data_augmentation.md`, `tfm_benchmark.md`).

**Current best: `config/experiments/final.yaml`** — OOF **0.9641**, public LB
**0.9640** (~median, rank ~1,181 / 2,398). Combines: per-class threshold tuning
(+0.0070), 5-fold / 1000 estimators (+0.0039), original-data augmentation
(+0.0004), interaction features (+0.0002), seed averaging, target encoding,
polynomial/log/ratio transforms, `simple_average` meta, pseudo-labeling
(threshold 0.95). Run with:

```bash
make train CONFIG=config/experiments/final.yaml RUN_NAME=final
```

Already-explored (diminishing returns, don't redo): categorical encoding
strategies (OHE/label/native all cluster 0.9524–0.9526), interaction features
*with* OHE categoricals (hurt the stack — use interactions only when
categoricals are dropped), and HP tuning (per-fold gains don't propagate through
the LogisticRegression meta). Open directions noted in `Iteration.md` Phase 1–5:
model diversity, tighter pseudo-labeling thresholds, GBM/weighted-average
meta-models, feature selection, and TFM blending as an alternative stack.
