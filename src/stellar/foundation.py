from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stellar.features import ColorFeatureEngineer

_DEFAULT_RAW_DROP = [
    "id",
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "fiber_ID",
    "plate",
    "MJD",
]


def build_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    feature_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test features for external tabular benchmark backends."""
    mode = feature_cfg.get("mode", "raw")
    drop_cols = feature_cfg.get("drop_cols", _DEFAULT_RAW_DROP)
    target_col = feature_cfg.get("target_col", "class")

    X_train = train.drop(columns=[target_col], errors="ignore").copy()
    X_test = test.copy()

    if mode == "raw":
        return (
            X_train.drop(columns=[c for c in drop_cols if c in X_train.columns], errors="ignore"),
            X_test.drop(columns=[c for c in drop_cols if c in X_test.columns], errors="ignore"),
        )

    if mode not in {"domain", "engineered"}:
        raise ValueError(f"Unknown feature mode: {mode!r}")

    encoding = feature_cfg.get("encoding", "passthrough" if mode == "domain" else "ohe")
    engineer = ColorFeatureEngineer(
        drop_cols=drop_cols,
        color_pairs=[tuple(p) for p in feature_cfg.get("color_pairs", [])],
        cat_cols=feature_cfg.get("cat_cols"),
        encoding=encoding,
        interaction_pairs=[tuple(p) for p in feature_cfg.get("interaction_pairs", [])],
        ratio_pairs=[tuple(p) for p in feature_cfg.get("ratio_pairs", [])],
        log_transform_cols=feature_cfg.get("log_transform_cols"),
        poly_cols=feature_cfg.get("poly_cols"),
        polynomial_degree=feature_cfg.get("polynomial_degree"),
    )
    need_y = encoding == "target"
    return (
        engineer.fit_transform(X_train, y_train if need_y else None),
        engineer.transform(X_test),
    )


def stratified_context_indices(
    y: pd.Series | np.ndarray,
    n_context: int | None,
    random_state: int = 42,
) -> np.ndarray:
    """Return stratified row positions for in-context models."""
    y_arr = np.asarray(y)
    n_total = len(y_arr)
    if n_context is None or n_context >= n_total:
        return np.arange(n_total)
    if n_context <= 0:
        raise ValueError("n_context must be positive")

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_arr, return_counts=True)
    proportions = counts / counts.sum()
    raw = proportions * n_context
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, counts)

    if n_context >= len(classes):
        quotas = np.maximum(quotas, 1)

    while quotas.sum() > n_context:
        candidates = np.where(quotas > 1)[0]
        if len(candidates) == 0:
            candidates = np.where(quotas > 0)[0]
        idx = candidates[np.argmin(raw[candidates] - quotas[candidates])]
        quotas[idx] -= 1

    while quotas.sum() < n_context:
        capacity = counts - quotas
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        idx = candidates[np.argmax(raw[candidates] - quotas[candidates])]
        quotas[idx] += 1

    selected: list[np.ndarray] = []
    for label, quota in zip(classes, quotas):
        if quota <= 0:
            continue
        positions = np.flatnonzero(y_arr == label)
        selected.append(rng.choice(positions, size=quota, replace=False))

    out = np.concatenate(selected) if selected else np.array([], dtype=int)
    rng.shuffle(out)
    return out


class BenchmarkBackend:
    classes_: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class DummyBackend(BenchmarkBackend):
    """Lightweight sklearn backend for offline smoke tests."""

    def __init__(self, params: dict[str, Any] | None = None):
        params = params.copy() if params else {}
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", 42)
        self.model = LogisticRegression(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class AutoGluonBackend(BenchmarkBackend):
    def __init__(self, params: dict[str, Any], run_dir: Path):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Backend 'autogluon' requires an optional package. "
                "Install the TFM dependency group with: uv sync --no-dev --group tfm"
            ) from exc

        self.TabularPredictor = TabularPredictor
        self.params = params.copy()
        self.run_dir = run_dir

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        label = self.params.get("label", "class")
        train_df = X.copy()
        train_df[label] = pd.Series(y, index=X.index).astype(str).astype(object)
        predictor_kwargs = self.params.get("predictor_kwargs", {})
        fit_kwargs = self.params.get("fit_kwargs", {})
        self.predictor = self.TabularPredictor(
            label=label,
            path=str(self.run_dir),
            **predictor_kwargs,
        )
        self.predictor.fit(train_df, **fit_kwargs)

        proba = self.predictor.predict_proba(X.head(1))
        self.classes_ = [str(c) for c in proba.columns]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict_proba(X).to_numpy()


class TabICLBackend(BenchmarkBackend):
    def __init__(self, params: dict[str, Any]):
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Backend 'tabicl' requires an optional package. "
                "Install the TFM dependency group with: uv sync --no-dev --group tfm"
            ) from exc

        self.model = TabICLClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class TabPFNBackend(BenchmarkBackend):
    def __init__(self, params: dict[str, Any]):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise RuntimeError(
                "Backend 'tabpfn' requires an optional package. "
                "Install the TFM dependency group with: uv sync --no-dev --group tfm"
            ) from exc

        params = params.copy()
        model_version = params.pop("model_version", None)
        if model_version:
            try:
                from tabpfn.constants import ModelVersion
            except ImportError as exc:
                raise RuntimeError("Installed tabpfn package does not expose ModelVersion") from exc

            version = getattr(ModelVersion, model_version)
            self.model = TabPFNClassifier.create_default_for_version(version)
            if params:
                self.model.set_params(**params)
        else:
            self.model = TabPFNClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


def make_backend(
    backend_cfg: dict[str, Any],
    run_dir: str | Path,
) -> BenchmarkBackend:
    name = backend_cfg["name"]
    params = backend_cfg.get("params", {})
    path = Path(run_dir)

    if name == "dummy":
        return DummyBackend(params)
    if name == "autogluon":
        return AutoGluonBackend(params, path)
    if name == "tabicl":
        return TabICLBackend(params)
    if name == "tabpfn":
        return TabPFNBackend(params)

    raise ValueError(f"Unknown benchmark backend: {name!r}")
