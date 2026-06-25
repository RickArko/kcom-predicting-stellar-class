"""Tests for data loading and augmentation."""

from __future__ import annotations

import pandas as pd

from stellar.data import load_data


class TestLoadDataAugmentation:
    def test_no_augment_returns_train_test_unchanged(self, tmp_path):
        train_df = pd.DataFrame({"id": [1, 2], "u": [20.0, 21.0], "class": ["GALAXY", "STAR"]})
        test_df = pd.DataFrame({"id": [3], "u": [22.0]})
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)

        train, test = load_data(str(tmp_path))
        assert len(train) == 2
        assert len(test) == 1

    def test_augment_concatenates_original(self, tmp_path):
        train_df = pd.DataFrame(
            {"id": [1, 2], "alpha": [1.0, 2.0], "u": [20.0, 21.0], "class": ["GALAXY", "STAR"]}
        )
        test_df = pd.DataFrame({"id": [3], "alpha": [3.0], "u": [22.0]})
        orig_df = pd.DataFrame(
            {"id": [10, 11], "alpha": [10.0, 11.0], "u": [30.0, 31.0], "class": ["QSO", "GALAXY"]}
        )
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)
        orig_df.to_csv(tmp_path / "original.csv", index=False)

        train, test = load_data(str(tmp_path), augment_path=str(tmp_path / "original.csv"))
        assert len(train) == 4
        assert len(test) == 1

    def test_augment_dedup_removes_duplicates(self, tmp_path):
        train_df = pd.DataFrame(
            {"id": [1, 2], "alpha": [1.0, 2.0], "u": [20.0, 21.0], "class": ["GALAXY", "STAR"]}
        )
        test_df = pd.DataFrame({"id": [3], "alpha": [3.0], "u": [22.0]})
        orig_df = pd.DataFrame({"id": [10], "alpha": [1.0], "u": [20.0], "class": ["GALAXY"]})
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)
        orig_df.to_csv(tmp_path / "original.csv", index=False)

        train, test = load_data(
            str(tmp_path),
            augment_path=str(tmp_path / "original.csv"),
            dedup_cols=["alpha", "u"],
        )
        assert len(train) == 2
        assert 1.0 in train["alpha"].values

    def test_augment_with_extra_columns_in_original(self, tmp_path):
        train_df = pd.DataFrame({"id": [1], "alpha": [1.0], "u": [20.0], "class": ["GALAXY"]})
        test_df = pd.DataFrame({"id": [3], "alpha": [3.0], "u": [22.0]})
        orig_df = pd.DataFrame(
            {
                "obj_ID": [100],
                "alpha": [10.0],
                "u": [30.0],
                "plate": [5000],
                "class": ["QSO"],
            }
        )
        train_df.to_csv(tmp_path / "train.csv", index=False)
        test_df.to_csv(tmp_path / "test.csv", index=False)
        orig_df.to_csv(tmp_path / "original.csv", index=False)

        train, test = load_data(str(tmp_path), augment_path=str(tmp_path / "original.csv"))
        assert len(train) == 2
        assert "obj_ID" in train.columns
        assert "plate" in train.columns
