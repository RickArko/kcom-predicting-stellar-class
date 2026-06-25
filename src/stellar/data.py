from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(
    data_dir: str = "data/",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    augment_path: str | None = None,
    dedup_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / train_file)
    test = pd.read_csv(data_path / test_file)

    if augment_path:
        orig = pd.read_csv(augment_path)
        before = len(train)
        train = pd.concat([train, orig], ignore_index=True)
        if dedup_cols:
            present = [c for c in dedup_cols if c in train.columns]
            train = train.drop_duplicates(subset=present, keep="first")
        after = len(train)
        logger.info(
            "Augmented with %s: %d → %d rows (%d added, %d deduped)",
            augment_path,
            before,
            after,
            len(orig),
            before + len(orig) - after,
        )

    return train, test
