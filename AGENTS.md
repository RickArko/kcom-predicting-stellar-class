# AGENTS.md — kcom-predicting-stellar-class

Kaggle Playground S6E6 — classify SDSS objects as **GALAXY / STAR / QSO**.
Metric: **balanced accuracy**. Deadline: **June 30, 2026**.

## Commands

All Python invocations **must** be prefixed with `uv run` (uv manages the env;
`.venv` exists but is not on PATH — bare `python`/`pytest` will fail or hit the
wrong interpreter).

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev` + editable install + kaggle auth check |
| `make download` | Fetch/expand competition CSVs into `data/` (requires auth) |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train ensemble & save a run |
| `make train ARGS="--flag x"` | Pass extra args to `scripts/train.py` |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run pytest tests/ -v` |
| `make test ARGS="-k name -x"` | Focused test run |
| `make lint` | `ruff check src/ scripts/ tests/` (check only, no autofix) |
| `make format` | `ruff format ... --check` (check only — does NOT rewrite) |
| `make format-fix` | Apply ruff formatting |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle + show leaderboard |

Verification loop after changes: `make lint && make format && make test`.
There is no typechecker or mypy configured.

## Architecture

- **Package**: `src/stellar/` — `data.py` (loaders), `features.py`
  (`ColorFeatureEngineer`), `models.py` (`StackingEnsemble`), `tracking.py`
  (run logger). Installed editable via hatchling (`packages = ["src/stellar"]`).
- **Ensemble**: LGBM + XGBoost + CatBoost base models → LogisticRegression
  meta-model on out-of-fold probabilities. Stratified k-fold CV.
- **Config-driven**: YAML in `config/` controls features, CV, and every model
  hyperparam. `config/config.yaml` = tuned default (5-fold, 1000 estimators);
  `config/baseline.yaml` = fast reference (3-fold, 250 estimators).
- **Experiments** live in `config/experiments/` — copy an existing config to
  start a new one.
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` containing a
  frozen `config.yaml`, `metrics.json`, `models/ensemble.joblib`, and
  `submission.csv`. Also writes the canonical `outputs/submissions/submission.csv`.
- **Re-predict without retraining**: `uv run python scripts/predict.py --run-dir outputs/runs/<name>`
- **Compare runs**: `uv run python scripts/compare.py` (reads `outputs/runs/`).

## Conventions & gotchas

- `ColorFeatureEngineer` `encoding` param: `"ohe"` (default), `"label"`
  (LabelEncoder → int32), `"passthrough"` (raw strings). Set via
  `features.encoding` in config.
- `StackingEnsemble.fit()` accepts `model_fit_kwargs` (dict keyed by model name)
  for per-model `fit()` kwargs that can't survive `sklearn.base.clone`.
  `scripts/train.py` auto-extracts `cat_features` from the `catboost` config
  section and passes it as a fit kwarg — do not pass `cat_features` to the
  CatBoost constructor.
- **LightGBM ≥4.6 ignores `categorical_feature`** in the sklearn wrapper
  constructor (warns "will be ignored"). Don't rely on it; use OHE or convert
  columns to `pd.Categorical` for native handling.
- `catboost_info/` is written during training and is gitignored — safe to delete.
- `from __future__ import annotations` is used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- Runtime: Python 3.13 (`.python-version`); `requires-python = ">=3.11"`.

## Testing

- Tests use **synthetic SDSS-like data** (the `synthetic_data` fixture in
  `tests/test_integration.py`) — `make test` works offline, no Kaggle download
  needed.
- `tests/test_models.py` covers the feature transformer, ensemble, and
  save/load roundtrip; `tests/test_integration.py` covers the end-to-end
  pipeline + submission format.

## Kaggle auth

- Token file: `.kaggle/access_token` (chmod 600), or env var `KAGGLE_API_TOKEN`.
  `make download` / `make submit` read the file first, then fall back to the env
  var. Legacy `~/.kaggle/kaggle.json` also works.
- You must **join the competition** (Accept Rules) on the Kaggle web page before
  `make download` works — otherwise it 403s.

## Gitignored artifacts (won't appear in `git status`)

`data/*.csv` and `*.zip`, `outputs/submissions/*` (except `.gitkeep`),
`outputs/runs/`, `models/`, `*.joblib`/`*.pkl`/`*.cbm`, `catboost_info/`,
`.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`, caches. Run data lives only
on disk — re-run `make download` on a fresh clone.

## Experiment state

Before running new experiments, read **`Iteration.md`** (full log + takeaways)
and **`docs/experiments/categoricals.md`** (encoding report). Headline: OHE
categoricals (`config/experiments/v001_keep_categoricals.yaml`) are the current
best at OOF **0.9526**; all categorical strategies cluster at 0.9524–0.9526 —
diminishing returns. Likely-better directions noted in `Iteration.md`:
interaction features (`redshift × colour`), target encoding, 5-fold final runs,
meta-model variants, feature selection, probability calibration.
