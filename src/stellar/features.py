"""Feature engineering for the Predicting Stellar Class competition."""

from __future__ import annotations

import pandas as pd

# Columns that carry no predictive signal (IDs and scan metadata)
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

# Photometric band pairs used to derive colour indices
_DEFAULT_COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "class",
    drop_cols: list[str] | None = None,
    color_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Apply feature engineering and return model-ready arrays.

    Steps
    -----
    1. Separate the target column from training features.
    2. Drop low-signal ID / metadata columns.
    3. Derive photometric colour indices (band differences).

    Parameters
    ----------
    train:
        Raw training DataFrame (must contain *target_col*).
    test:
        Raw test DataFrame (no target column).
    target_col:
        Name of the target column in *train*.
    drop_cols:
        Columns to drop from both DataFrames.  Defaults to the standard list
        of scan / ID columns.
    color_pairs:
        Band pairs to subtract.  Defaults to (u-g, g-r, r-i, i-z).

    Returns
    -------
    X_train, X_test, y_train
    """
    if drop_cols is None:
        drop_cols = _DEFAULT_DROP
    if color_pairs is None:
        color_pairs = _DEFAULT_COLOR_PAIRS

    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col], errors="ignore").copy()
    X_test = test.copy()

    # Drop metadata columns present in each DataFrame
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Derive colour indices
    for band_a, band_b in color_pairs:
        if band_a in X_train.columns and band_b in X_train.columns:
            col_name = f"{band_a}_{band_b}"
            X_train[col_name] = X_train[band_a] - X_train[band_b]
            X_test[col_name] = X_test[band_a] - X_test[band_b]

    return X_train, X_test, y_train
