"""Unit tests for the stellar classification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stellar.data import decode_target, encode_target
from stellar.models import save_submission


class TestEncodeTarget:
    def test_basic_encoding(self):
        s = pd.Series(["STAR", "GALAXY", "QSO", "GALAXY", "STAR"])
        encoded, label_map = encode_target(s)
        assert label_map == {"GALAXY": 0, "QSO": 1, "STAR": 2}
        assert encoded.tolist() == [2, 0, 1, 0, 2]

    def test_decode_roundtrip(self):
        s = pd.Series(["STAR", "GALAXY", "QSO"])
        encoded, label_map = encode_target(s)
        decoded = decode_target(encoded, label_map)
        assert decoded.tolist() == s.tolist()


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
