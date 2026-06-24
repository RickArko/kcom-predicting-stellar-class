"""End-to-end integration test with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stellar.features import make_features
from stellar.models import save_submission, train_cv


@pytest.fixture
def synthetic_data():
    n_train, n_test = 200, 50
    rng = np.random.default_rng(42)

    def _make_df(n, has_target=False):
        df = pd.DataFrame({
            "id": range(n),
            "alpha": rng.uniform(0, 360, n),
            "delta": rng.uniform(-90, 90, n),
            "u": rng.uniform(15, 25, n),
            "g": rng.uniform(14, 24, n),
            "r": rng.uniform(13, 23, n),
            "i": rng.uniform(12, 22, n),
            "z": rng.uniform(11, 21, n),
            "redshift": rng.exponential(0.5, n),
            "obj_ID": rng.integers(1e6, 1e7, n),
            "run_ID": rng.integers(1, 100, n),
            "rerun_ID": rng.integers(1, 10, n),
            "cam_col": rng.integers(1, 6, n),
            "field_ID": rng.integers(1, 1000, n),
            "spec_obj_ID": rng.integers(1e6, 1e7, n),
            "fiber_ID": rng.integers(1, 1000, n),
        })
        if has_target:
            df["class"] = rng.choice(["GALAXY", "STAR", "QSO"], n)
        return df

    train = _make_df(n_train, has_target=True)
    test = _make_df(n_test, has_target=False)
    return train, test


class TestPipeline:
    def test_feature_engineering(self, synthetic_data):
        train, test = synthetic_data
        X_train, X_test, y_train = make_features(train, test)
        assert "u_g" in X_train.columns
        assert "g_r" in X_train.columns
        assert "r_i" in X_test.columns
        assert "i_z" in X_test.columns
        assert all(c not in X_train.columns for c in ["obj_ID", "run_ID", "cam_col"])
        assert len(X_train) == len(train)
        assert len(X_test) == len(test)

    def test_train_cv_returns_valid_predictions(self, synthetic_data):
        train, test = synthetic_data
        X_train, X_test, y_train = make_features(train, test)
        oof_preds, test_preds = train_cv(
            X_train, y_train, X_test,
            n_splits=2,
            random_state=42,
        )
        assert len(test_preds) == len(test)
        assert set(test_preds) <= {"GALAXY", "STAR", "QSO"}
        assert oof_preds.shape == (len(X_train), 3 * 3)

    def test_full_pipeline_with_submission(self, synthetic_data, tmp_path):
        train, test = synthetic_data
        X_train, X_test, y_train = make_features(train, test)
        _, test_preds = train_cv(
            X_train, y_train, X_test,
            n_splits=2,
            random_state=42,
        )
        out = tmp_path / "sub.csv"
        save_submission(test["id"], test_preds, output_path=str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "class"]
        assert len(df) == len(test)
        assert all(c in {"GALAXY", "STAR", "QSO"} for c in df["class"])
