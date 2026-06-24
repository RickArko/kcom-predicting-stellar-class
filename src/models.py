"""Model training, cross-validation, and ensembling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def _get_lgbm(params: dict | None = None):
    from lightgbm import LGBMClassifier

    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def _get_xgb(params: dict | None = None):
    from xgboost import XGBClassifier

    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def _get_catboost(params: dict | None = None):
    from catboost import CatBoostClassifier

    defaults = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
    )
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


def train_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    model_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM, XGBoost, and CatBoost with stratified k-fold CV and
    combine predictions using a Logistic Regression meta-model (stacking).

    Parameters
    ----------
    X_train, y_train:
        Training features and string target labels.
    X_test:
        Test features.
    n_splits:
        Number of CV folds.
    random_state:
        Random seed for reproducibility.
    model_params:
        Optional dict with keys ``lgbm``, ``xgb``, ``catboost`` to override
        default hyper-parameters for each base learner.

    Returns
    -------
    oof_preds:
        Out-of-fold predictions (shape: n_train × n_classes).
    test_preds:
        Final test set class predictions as a 1-D array of string labels.
    """
    model_params = model_params or {}

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    base_models = [
        ("lgbm", _get_lgbm(model_params.get("lgbm"))),
        ("xgb", _get_xgb(model_params.get("xgb"))),
        ("catboost", _get_catboost(model_params.get("catboost"))),
    ]

    # Arrays to hold OOF and test probability predictions from each base model
    oof_proba = {name: np.zeros((len(X_train), n_classes)) for name, _ in base_models}
    test_proba = {name: np.zeros((len(X_test), n_classes)) for name, _ in base_models}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_enc), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        for name, model in base_models:
            model.fit(X_tr, y_tr)
            oof_proba[name][val_idx] = model.predict_proba(X_val)
            test_proba[name] += model.predict_proba(X_test) / n_splits

        fold_preds = np.argmax(
            sum(oof_proba[n][val_idx] for n, _ in base_models), axis=1
        )
        fold_score = balanced_accuracy_score(y_val, fold_preds)
        print(f"Fold {fold} balanced accuracy: {fold_score:.4f}")

    # Stack OOF probabilities as meta-features
    oof_meta = np.hstack([oof_proba[n] for n, _ in base_models])
    test_meta = np.hstack([test_proba[n] for n, _ in base_models])

    meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
    meta_model.fit(oof_meta, y_enc)

    oof_final = meta_model.predict(oof_meta)
    print(
        f"\nOOF balanced accuracy (stacked): "
        f"{balanced_accuracy_score(y_enc, oof_final):.4f}"
    )

    test_final_enc = meta_model.predict(test_meta)
    test_preds = le.inverse_transform(test_final_enc)

    return oof_meta, test_preds


def save_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = "outputs/submissions/submission.csv",
) -> None:
    """Write a submission CSV in the required format (id, class).

    Parameters
    ----------
    test_ids:
        The ``id`` column from the test DataFrame.
    predictions:
        1-D array of string class labels (GALAXY, STAR, QSO).
    output_path:
        Path where the CSV will be saved.
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "class": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
