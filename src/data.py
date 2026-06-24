"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(
    data_dir: str = "data/",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets.

    Parameters
    ----------
    data_dir:
        Directory containing the CSV files.
    train_file:
        Filename for the training set.
    test_file:
        Filename for the test set.

    Returns
    -------
    train, test:
        Raw DataFrames for the train and test sets.
    """
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / train_file)
    test = pd.read_csv(data_path / test_file)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test


def encode_target(series: pd.Series) -> tuple[pd.Series, dict]:
    """Encode string class labels to integers.

    Parameters
    ----------
    series:
        Target column with string class labels (GALAXY, STAR, QSO).

    Returns
    -------
    encoded:
        Integer-encoded target series.
    label_map:
        Mapping from string label to integer (e.g. {"GALAXY": 0, ...}).
    """
    classes = sorted(series.unique())
    label_map = {cls: i for i, cls in enumerate(classes)}
    return series.map(label_map), label_map


def decode_target(series: pd.Series, label_map: dict) -> pd.Series:
    """Decode integer predictions back to string class labels.

    Parameters
    ----------
    series:
        Integer-encoded predictions.
    label_map:
        Mapping returned by :func:`encode_target`.

    Returns
    -------
    pd.Series
        String class labels.
    """
    inverse = {v: k for k, v in label_map.items()}
    return series.map(inverse)
