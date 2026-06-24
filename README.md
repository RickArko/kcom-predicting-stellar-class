# kcom-predicting-stellar-class

Kaggle Competition: [Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6) (Playground Series S6E6)

## Overview

Classify astronomical objects from the Sloan Digital Sky Survey (SDSS) into one of three categories:

- **GALAXY** – Extended sources (galaxies)
- **STAR** – Point sources (stars)
- **QSO** – Quasi-stellar objects (quasars)

**Evaluation Metric:** Balanced Accuracy  
**Deadline:** June 30, 2026

## Dataset Features

| Feature | Description |
|---------|-------------|
| `alpha` | Right Ascension (J2000 epoch) |
| `delta` | Declination (J2000 epoch) |
| `u`, `g`, `r`, `i`, `z` | Photometric magnitudes (UV → Infrared) |
| `redshift` | Redshift value (key discriminator) |
| `obj_ID` | Unique SDSS object identifier |
| `run_ID`, `rerun_ID`, `cam_col`, `field_ID` | Scan metadata |
| `spec_obj_ID`, `fiber_ID`, `plate`, `MJD` | Spectroscopic metadata |
| `class` | **Target** – GALAXY, STAR, or QSO |

## Approach

1. **EDA** – Distribution analysis, correlation heatmaps, class balance
2. **Feature Engineering** – Photometric color indices (`u-g`, `g-r`, `r-i`, `i-z`), drop low-signal columns
3. **Modeling** – LightGBM, XGBoost, CatBoost with stratified k-fold CV
4. **Ensemble** – Stacking with Logistic Regression meta-model

## Repository Structure

```
kcom-predicting-stellar-class/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── config.yaml          # Experiment configuration
├── data/                    # Not tracked by git — download from Kaggle
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading & preprocessing
│   ├── features.py          # Feature engineering
│   └── models.py            # Model training & evaluation
└── outputs/
    └── submissions/         # Generated submission CSVs
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (requires Kaggle API token)
kaggle competitions download -c playground-series-s6e6 -p data/
unzip data/playground-series-s6e6.zip -d data/
```

## Quick Start

```python
from src.data import load_data
from src.features import make_features
from src.models import train_cv

train, test = load_data("data/")
X_train, X_test, y_train = make_features(train, test)
oof_preds, test_preds = train_cv(X_train, y_train, X_test)
```
