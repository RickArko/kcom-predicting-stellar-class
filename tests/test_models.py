"""Unit tests for core components using synthetic data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from stellar.features import ColorFeatureEngineer, make_features
from stellar.models import StackingEnsemble, save_submission


class TestColorFeatureEngineer:
    def test_drops_metadata_columns(self):
        X = pd.DataFrame(
            {
                "id": [1, 2],
                "obj_ID": [100, 200],
                "run_ID": [1, 2],
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "r": [18.0, 19.0],
            }
        )
        engineer = ColorFeatureEngineer()
        out = engineer.fit_transform(X)
        assert "id" not in out.columns
        assert "obj_ID" not in out.columns
        assert "run_ID" not in out.columns
        assert "u" in out.columns
        assert "g" in out.columns

    def test_adds_color_indices(self):
        X = pd.DataFrame(
            {
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "r": [18.0, 19.0],
                "i": [17.0, 18.0],
                "z": [16.0, 17.0],
            }
        )
        engineer = ColorFeatureEngineer()
        out = engineer.fit_transform(X)
        assert "u_g" in out.columns
        assert "g_r" in out.columns
        assert "r_i" in out.columns
        assert "i_z" in out.columns
        np.testing.assert_array_almost_equal(out["u_g"], [1.0, 1.0])
        np.testing.assert_array_almost_equal(out["g_r"], [1.0, 1.0])

    def test_adds_interaction_features(self):
        X = pd.DataFrame(
            {
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "r": [18.0, 19.0],
                "i": [17.0, 18.0],
                "z": [16.0, 17.0],
                "redshift": [0.1, 0.2],
            }
        )
        engineer = ColorFeatureEngineer(
            interaction_pairs=[("redshift", "u_g"), ("u_g", "g_r")],
        )
        out = engineer.fit_transform(X)
        assert "redshift_x_u_g" in out.columns
        assert "u_g_x_g_r" in out.columns
        np.testing.assert_array_almost_equal(out["redshift_x_u_g"], [0.1, 0.2])
        np.testing.assert_array_almost_equal(out["u_g_x_g_r"], [1.0, 1.0])

    def test_invalid_interaction_pair_dropped(self):
        X = pd.DataFrame(
            {
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "r": [18.0, 19.0],
                "i": [17.0, 18.0],
                "z": [16.0, 17.0],
            }
        )
        engineer = ColorFeatureEngineer(
            interaction_pairs=[("redshift", "u_g"), ("u_g", "g_r")],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = engineer.fit_transform(X)
        assert "redshift_x_u_g" not in out.columns
        assert "u_g_x_g_r" in out.columns

    def test_consistent_columns_train_test(self):
        train = pd.DataFrame(
            {
                "id": [1, 2],
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "r": [18.0, 19.0],
                "i": [17.0, 18.0],
                "z": [16.0, 17.0],
            }
        )
        test = pd.DataFrame(
            {
                "id": [3, 4],
                "u": [22.0, 23.0],
                "g": [21.0, 22.0],
                "r": [20.0, 21.0],
                "i": [19.0, 20.0],
                "z": [18.0, 19.0],
            }
        )
        engineer = ColorFeatureEngineer()
        X_train = engineer.fit_transform(train)
        X_test = engineer.transform(test)
        assert list(X_train.columns) == list(X_test.columns)

    def test_make_features_legacy_wrapper(self):
        train = pd.DataFrame(
            {
                "id": [1, 2],
                "u": [20.0, 21.0],
                "g": [19.0, 20.0],
                "class": ["GALAXY", "STAR"],
            }
        )
        test = pd.DataFrame(
            {
                "id": [3],
                "u": [22.0],
                "g": [21.0],
            }
        )
        X_train, X_test, y_train = make_features(train, test)
        assert "class" not in X_train.columns
        assert len(y_train) == 2
        assert list(y_train) == ["GALAXY", "STAR"]
        assert "u_g" in X_train.columns


class TestStackingEnsemble:
    def test_fit_and_predict_on_synthetic_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(
            {
                "u": rng.uniform(15, 25, n),
                "g": rng.uniform(14, 24, n),
                "r": rng.uniform(13, 23, n),
                "i": rng.uniform(12, 22, n),
                "z": rng.uniform(11, 21, n),
            }
        )
        classes = ["GALAXY", "STAR", "QSO"]
        y = pd.Series(rng.choice(classes, n))

        engineer = ColorFeatureEngineer()
        X_feat = engineer.fit_transform(X)

        base_models = [
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ]
        ensemble = StackingEnsemble(base_models)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ensemble.fit(X_feat, y, cv)

        assert len(ensemble.valid_scores_) == 3
        assert ensemble.overall_oof_score_ is not None
        assert all(0 <= s <= 1 for s in ensemble.valid_scores_)

        preds = ensemble.predict(X_feat)
        assert len(preds) == n
        assert set(preds) <= set(classes)

    def test_save_load_roundtrip(self, tmp_path):
        n = 50
        X = pd.DataFrame(
            {
                "u": [20.0] * n,
                "g": [19.0] * n,
                "r": [18.0] * n,
                "i": [17.0] * n,
                "z": [16.0] * n,
            }
        )
        y = pd.Series(["GALAXY", "STAR"] * (n // 2))
        engineer = ColorFeatureEngineer()
        X_feat = engineer.fit_transform(X)

        ensemble = StackingEnsemble([("lr", LogisticRegression(max_iter=500, random_state=42))])
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        ensemble.fit(X_feat, y, cv)

        model_path = tmp_path / "ensemble.joblib"
        ensemble.save(model_path)
        assert model_path.exists()

        loaded = StackingEnsemble.load(model_path)
        preds_orig = ensemble.predict(X_feat)
        preds_loaded = loaded.predict(X_feat)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSaveSubmission:
    def test_saves_correct_format(self, tmp_path):
        ids = pd.Series([0, 1, 2])
        preds = np.array(["STAR", "GALAXY", "QSO"])
        out = tmp_path / "sub.csv"
        save_submission(ids, preds, output_path=str(out))
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "class"]
        assert df["class"].tolist() == ["STAR", "GALAXY", "QSO"]
        assert df["id"].tolist() == [0, 1, 2]


class TestEncodeConsistency:
    def test_label_encoder_matches_data_loading(self):
        le = LabelEncoder()
        classes = le.fit_transform(["GALAXY", "STAR", "QSO", "GALAXY", "STAR"])
        assert list(classes) == [0, 2, 1, 0, 2]
        assert list(le.classes_) == ["GALAXY", "QSO", "STAR"]

    def test_roundtrip_with_model_prediction(self):
        le = LabelEncoder()
        y = ["GALAXY", "STAR", "QSO"]
        le.fit(y)
        inverse = le.inverse_transform([0, 1, 2])
        assert list(inverse) == ["GALAXY", "QSO", "STAR"]
