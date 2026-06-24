"""End-to-end integration test with structured synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from stellar.features import ColorFeatureEngineer
from stellar.models import StackingEnsemble, save_submission


@pytest.fixture
def synthetic_data():
    """Generate separable synthetic SDSS-like data.

    Classes are made partially separable by their (u, z) band positions
    so the pipeline has signal to learn from.
    """
    n_train, n_test = 300, 100
    rng = np.random.default_rng(42)

    def _make_df(n, has_target=False):
        df = pd.DataFrame(
            {
                "id": range(n),
                "u": rng.uniform(15, 25, n),
                "g": rng.uniform(14, 24, n),
                "r": rng.uniform(13, 23, n),
                "i": rng.uniform(12, 22, n),
                "z": rng.uniform(11, 21, n),
                "redshift": rng.exponential(0.5, n),
                "obj_ID": rng.integers(1e6, 1e7, n),
                "run_ID": rng.integers(1, 100, n),
                "cam_col": rng.integers(1, 6, n),
                "field_ID": rng.integers(1, 1000, n),
            }
        )
        if has_target:
            conditions = [
                (df["u"] > 20) & (df["z"] < 16),
                (df["u"] < 18) & (df["z"] > 18),
            ]
            df["class"] = np.select(conditions, ["GALAXY", "QSO"], default="STAR")
        return df

    train = _make_df(n_train, has_target=True)
    test = _make_df(n_test, has_target=False)
    return train, test


class TestPipeline:
    def test_feature_engineering(self, synthetic_data):
        train, test = synthetic_data
        engineer = ColorFeatureEngineer()
        X_train = engineer.fit_transform(train.drop(columns=["class"]))
        X_test = engineer.transform(test)

        assert "u_g" in X_train.columns
        assert "g_r" in X_train.columns
        assert "r_i" in X_test.columns
        assert "i_z" in X_test.columns
        for c in ["obj_ID", "run_ID", "cam_col"]:
            assert c not in X_train.columns
            assert c not in X_test.columns
        assert len(X_train) == len(train)
        assert len(X_test) == len(test)
        assert X_train.columns.tolist() == X_test.columns.tolist()

    def test_ensemble_learns_from_data(self, synthetic_data):
        train, test = synthetic_data
        engineer = ColorFeatureEngineer()
        y_train = train["class"]
        X_train = engineer.fit_transform(train.drop(columns=["class"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr1", LogisticRegression(max_iter=1000, random_state=42)),
            ("lr2", LogisticRegression(max_iter=1000, random_state=42, C=0.5)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        assert len(ensemble.valid_scores_) == 3
        assert ensemble.overall_oof_score_ is not None
        assert len(ensemble.fold_models_) == 3

        preds = ensemble.predict(X_test)
        assert len(preds) == len(test)
        assert set(preds) <= {"GALAXY", "STAR", "QSO"}

    def test_full_pipeline_with_submission(self, synthetic_data, tmp_path):
        train, test = synthetic_data
        engineer = ColorFeatureEngineer()
        y_train = train["class"]
        X_train = engineer.fit_transform(train.drop(columns=["class"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        preds = ensemble.predict(X_test)
        out = tmp_path / "sub.csv"
        save_submission(test["id"], preds, output_path=str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "class"]
        assert len(df) == len(test)
        assert all(c in {"GALAXY", "STAR", "QSO"} for c in df["class"])

    def test_save_load_and_predict_consistent(self, synthetic_data, tmp_path):
        train, test = synthetic_data
        engineer = ColorFeatureEngineer()
        y_train = train["class"]
        X_train = engineer.fit_transform(train.drop(columns=["class"]))
        X_test = engineer.transform(test)

        base_models = [
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_train, y_train, cv, X_test)

        model_path = tmp_path / "ensemble.joblib"
        ensemble.save(model_path)

        loaded = StackingEnsemble.load(model_path)
        preds_orig = ensemble.predict(X_test)
        preds_loaded = loaded.predict(X_test)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
