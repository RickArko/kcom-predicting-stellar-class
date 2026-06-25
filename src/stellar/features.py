from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

_DEFAULT_DROP = [
    "id",
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "fiber_ID",
    "spectral_type",
    "galaxy_population",
    "plate",
    "MJD",
]

_DEFAULT_COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]


class ColorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Drop low-signal columns, derive colour indices, encode categoricals.

    Parameters
    ----------
    drop_cols:
        Column names to remove.  Only columns actually present are dropped.
    color_pairs:
        Band pairs to subtract, each entry is ``(band_a, band_b)``.
    cat_cols:
        Column names to encode.  If ``None`` (default), any object-dtype
        column not in *drop_cols* is automatically encoded.
    encoding:
        How to encode categorical columns.
        ``"ohe"`` (default) — one-hot encode via ``OneHotEncoder``.
        ``"label"`` — ordinal label encode via ``LabelEncoder``.
        ``"passthrough"`` — keep raw string values unchanged.
    """

    def __init__(
        self,
        drop_cols: list[str] | None = None,
        color_pairs: list[tuple[str, str]] | None = None,
        cat_cols: list[str] | None = None,
        encoding: str = "ohe",
    ):
        self.drop_cols = drop_cols or _DEFAULT_DROP
        self.color_pairs = color_pairs or _DEFAULT_COLOR_PAIRS
        self.cat_cols = cat_cols
        self.encoding = encoding

    def fit(self, X: pd.DataFrame, y=None) -> ColorFeatureEngineer:
        self._drop_cols_ = [c for c in self.drop_cols if c in X.columns]
        self._color_pairs_ = [
            (a, b) for a, b in self.color_pairs if a in X.columns and b in X.columns
        ]
        if self.cat_cols is not None:
            self._cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        else:
            self._cat_cols_ = [
                c for c in X.columns if X[c].dtype == "object" and c not in self._drop_cols_
            ]

        if self.encoding == "ohe":
            self._encoders_: dict[str, OneHotEncoder] = {}
            for c in self._cat_cols_:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(X[[c]])
                self._encoders_[c] = enc
        elif self.encoding == "label":
            self._encoders_ = {}
            for c in self._cat_cols_:
                enc = LabelEncoder()
                enc.fit(X[c])
                self._encoders_[c] = enc
        elif self.encoding == "passthrough":
            self._encoders_ = {}
        else:
            raise ValueError(f"Unknown encoding: {self.encoding!r}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=self._drop_cols_, errors="ignore")
        for a, b in self._color_pairs_:
            X[f"{a}_{b}"] = X[a] - X[b]

        if self.encoding == "ohe":
            for c in self._cat_cols_:
                enc = self._encoders_[c]
                encoded = enc.transform(X[[c]])
                col_names = [f"{c}_{v}" for v in enc.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=col_names, index=X.index).astype(np.int8)
                X = pd.concat([X.drop(columns=[c]), encoded_df], axis=1)
        elif self.encoding == "label":
            for c in self._cat_cols_:
                enc = self._encoders_[c]
                X[c] = enc.transform(X[c])
                X[c] = X[c].astype(np.int32)
        elif self.encoding == "passthrough":
            pass

        return X


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "class",
    drop_cols: list[str] | None = None,
    color_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    engineer = ColorFeatureEngineer(drop_cols=drop_cols, color_pairs=color_pairs)
    y_train = train[target_col].copy()
    X_train = engineer.fit_transform(train.drop(columns=[target_col]))
    X_test = engineer.transform(test.copy())
    return X_train, X_test, y_train
