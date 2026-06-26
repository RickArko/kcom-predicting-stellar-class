from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures

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
        ``"target"`` — target-mean encode using the training target.
        ``"passthrough"`` — keep raw string values unchanged.
    interaction_pairs:
        Pairs of column names to multiply as interaction features.  Each
        entry is ``(col_a, col_b)`` producing a column ``"{a}_x_{b}"``.
        A pair is valid if both names survive *drop_cols* or are produced
        by *color_pairs* (e.g. ``"u_g"``).  Invalid pairs are dropped with
        a warning.  Created after colour indices and before encoding.
    ratio_pairs:
        Band pairs to divide, each entry is ``(band_a, band_b)`` producing
        ``"{a}_{b}_ratio"``.  Applied after colour indices.
    log_transform_cols:
        Column names to log-transform (via ``log1p``).  Produces
        ``"{col}_log"``.  Applied after colour indices.
    poly_cols:
        Numeric column names to expand with polynomial features.  Only
        used when *polynomial_degree* is set.  Applied after colour indices
        and before encoding.
    polynomial_degree:
        Degree for ``PolynomialFeatures``.  When set, *poly_cols* (or
        all numeric columns) are expanded with interaction/power terms.
    """

    def __init__(
        self,
        drop_cols: list[str] | None = None,
        color_pairs: list[tuple[str, str]] | None = None,
        cat_cols: list[str] | None = None,
        encoding: str = "ohe",
        interaction_pairs: list[tuple[str, str]] | None = None,
        ratio_pairs: list[tuple[str, str]] | None = None,
        log_transform_cols: list[str] | None = None,
        poly_cols: list[str] | None = None,
        polynomial_degree: int | None = None,
    ):
        self.drop_cols = _DEFAULT_DROP if drop_cols is None else drop_cols
        self.color_pairs = _DEFAULT_COLOR_PAIRS if color_pairs is None else color_pairs
        self.cat_cols = cat_cols
        self.encoding = encoding
        self.interaction_pairs = interaction_pairs
        self.ratio_pairs = ratio_pairs
        self.log_transform_cols = log_transform_cols if log_transform_cols is not None else []
        self.poly_cols = poly_cols
        self.polynomial_degree = polynomial_degree

    def fit(self, X: pd.DataFrame, y=None) -> ColorFeatureEngineer:
        self._drop_cols_ = [c for c in self.drop_cols if c in X.columns]
        self._color_pairs_ = [
            (a, b) for a, b in self.color_pairs if a in X.columns and b in X.columns
        ]

        available = {c for c in X.columns if c not in self._drop_cols_}
        available |= {f"{a}_{b}" for a, b in self._color_pairs_}

        self._interaction_pairs_ = []
        for a, b in self.interaction_pairs or []:
            if a in available and b in available:
                self._interaction_pairs_.append((a, b))
            else:
                warnings.warn(
                    f"interaction_pair ({a!r}, {b!r}) references unknown or "
                    f"dropped columns — skipping.",
                    stacklevel=2,
                )

        self._ratio_pairs_ = [
            (a, b) for a, b in (self.ratio_pairs or []) if a in X.columns and b in X.columns
        ]

        self._log_transform_cols_ = [c for c in self.log_transform_cols if c in X.columns]

        if self.polynomial_degree and self.poly_cols:
            self._poly_cols_ = [
                c for c in self.poly_cols if c in X.columns and c not in self._drop_cols_
            ]
            if self._poly_cols_:
                self._poly_ = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=False,
                    interaction_only=False,
                )
                self._poly_.fit(X[self._poly_cols_])
        else:
            self._poly_cols_ = []
            self._poly_ = None

        if self.cat_cols is not None:
            self._cat_cols_ = [c for c in self.cat_cols if c in X.columns]
        else:
            self._cat_cols_ = [
                c for c in X.columns if X[c].dtype == "object" and c not in self._drop_cols_
            ]

        if self.encoding == "target":
            if y is None:
                raise ValueError("y must be provided when encoding='target'")
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self._target_encoders_: dict[str, dict] = {}
            for c in self._cat_cols_:
                global_mean = float(y_enc.mean())
                stats = pd.DataFrame({"y": y_enc, "cat": X[c].values})
                grouped = stats.groupby("cat")["y"].agg(["mean", "count"])
                smoothing = 10.0
                smoothed = (grouped["mean"] * grouped["count"] + global_mean * smoothing) / (
                    grouped["count"] + smoothing
                )
                encoding = smoothed.to_dict()
                encoding["_global_mean_"] = global_mean
                self._target_encoders_[c] = encoding
        elif self.encoding == "ohe":
            self._encoders_ = {}
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

        for a, b in self._ratio_pairs_:
            ratio = X[a] / X[b]
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            X[f"{a}_{b}_ratio"] = ratio

        for c in self._log_transform_cols_:
            X[f"{c}_log"] = np.log1p(X[c].clip(lower=0))

        for a, b in self._interaction_pairs_:
            X[f"{a}_x_{b}"] = X[a] * X[b]

        if self._poly_ is not None and self._poly_cols_:
            poly_vals = self._poly_.transform(X[self._poly_cols_])
            poly_names = self._poly_.get_feature_names_out(self._poly_cols_)
            # LightGBM normalises spaces → underscores internally, so rename
            # interaction terms to avoid collisions (e.g. "u g" vs "u_g").
            poly_names = [n.replace(" ", "_x_") for n in poly_names]
            existing = set(X.columns)
            new_poly = [c for c in poly_names if c not in existing]
            if new_poly:
                idx = [list(poly_names).index(c) for c in new_poly]
                X_poly = pd.DataFrame(poly_vals[:, idx], columns=new_poly, index=X.index)
                X = pd.concat([X, X_poly], axis=1)

        if self.encoding == "target":
            for c in self._cat_cols_:
                enc = self._target_encoders_[c]
                global_mean = enc["_global_mean_"]
                mapping = {k: v for k, v in enc.items() if k != "_global_mean_"}
                X[c] = X[c].map(mapping).fillna(global_mean).astype(np.float32)
        elif self.encoding == "ohe":
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
    X_train = engineer.fit_transform(train.drop(columns=[target_col]), y_train)
    X_test = engineer.transform(test.copy())
    return X_train, X_test, y_train
