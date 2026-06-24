from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(
    data_dir: str = "data/",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / train_file)
    test = pd.read_csv(data_path / test_file)
    return train, test
