from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xgboost import XGBClassifier

MODEL_REGISTRY: dict[str, type] = {
    "lgbm": LGBMClassifier,
    "xgb": XGBClassifier,
    "catboost": CatBoostClassifier,
    "extratrees": ExtraTreesClassifier,
    "histgbm": HistGradientBoostingClassifier,
    "ridge": RidgeClassifier,
}

SEED_PARAM_MAP: dict[str, str] = {
    "lgbm": "random_state",
    "xgb": "random_state",
    "catboost": "random_seed",
    "extratrees": "random_state",
    "histgbm": "random_state",
    "ridge": "random_state",
}


class SimpleAverageMeta(BaseEstimator, ClassifierMixin):
    """Meta-model that averages base-model probabilities and takes argmax.

    No learning — just reshapes the stacked probabilities back to
    ``(n_models, n_samples, n_classes)``, averages across models, and
    returns the argmax.  Useful as a baseline to check whether the
    LogisticRegression meta-model is overfitting the OOF probabilities.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_models_ = X.shape[1] // len(self.classes_)
        return self

    def predict(self, X):
        n_classes = len(self.classes_)
        reshaped = X.reshape(X.shape[0], self.n_models_, n_classes)
        avg = reshaped.mean(axis=1)
        return np.argmax(avg, axis=1)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        reshaped = X.reshape(X.shape[0], self.n_models_, n_classes)
        return reshaped.mean(axis=1)


class WeightedAverageMeta(BaseEstimator, ClassifierMixin):
    """Learns per-model per-class weights via constrained optimization.

    Optimises balanced accuracy on the OOF meta-features by scaling each
    (model, class) pair's probability contribution.  Uses L-BFGS-B with
    non-negativity constraints so weights are interpretable.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_models = X.shape[1] // n_classes

        from scipy.optimize import minimize

        reshaped = X.reshape(X.shape[0], n_models, n_classes)

        def neg_balanced_acc(weights):
            w = weights.reshape(n_models, n_classes)
            scores = (reshaped * w[np.newaxis, :, :]).sum(axis=1)
            preds = np.argmax(scores, axis=1)
            return -balanced_accuracy_score(y, preds)

        x0 = np.ones(n_models * n_classes)
        bounds = [(0, None)] * (n_models * n_classes)
        result = minimize(neg_balanced_acc, x0, method="L-BFGS-B", bounds=bounds)
        self.weights_ = result.x.reshape(n_models, n_classes)
        return self

    def predict(self, X):
        n_classes = len(self.classes_)
        n_models = X.shape[1] // n_classes
        reshaped = X.reshape(X.shape[0], n_models, n_classes)
        scores = (reshaped * self.weights_[np.newaxis, :, :]).sum(axis=1)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        n_models = X.shape[1] // n_classes
        reshaped = X.reshape(X.shape[0], n_models, n_classes)
        scores = (reshaped * self.weights_[np.newaxis, :, :]).sum(axis=1)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return scores / row_sums


class StackingEnsemble:
    """Stacking ensemble with stratified k-fold CV and OOF meta-features.

    Trains a set of base models via cross-validation, collects out-of-fold
    probability predictions, and trains a Logistic Regression meta-model on
    those predictions for final calibration.

    Parameters
    ----------
    base_models:
        List of ``(name, estimator)`` tuples.  Each estimator must be
        sklearn-compatible (implement ``get_params`` / ``set_params`` so
        that ``sklearn.base.clone`` works).
    meta_model:
        The meta-learner.  Defaults to ``LogisticRegression(max_iter=1000)``.
    """

    def __init__(
        self,
        base_models: list[tuple[str, object]],
        meta_model: object | None = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=1000, random_state=42)
        self.fold_models_: list[dict[str, object]] = []
        self.meta_model_: object | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.n_classes_: int | None = None
        self.valid_scores_: list[float] = []
        self.overall_oof_score_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: object,
        X_test: pd.DataFrame | None = None,
        model_fit_kwargs: dict[str, dict] | None = None,
    ) -> StackingEnsemble:
        """Fit the stacking ensemble.

        Parameters
        ----------
        X:
            Training features.
        y:
            Training target (string class labels).
        cv:
            A cross-validator (e.g. ``StratifiedKFold``).
        X_test:
            Optional test set.  When provided, test-set probability
            predictions are computed during training (avoids re-running
            all fold models later for the competition test set).
        model_fit_kwargs:
            Optional dict mapping model name to keyword arguments passed
            to that model's ``fit()`` method.  Useful for models whose
            sklearn wrappers accept per-fit params (e.g. CatBoost's
            ``cat_features``) that cannot survive ``sklearn.base.clone``.

        Returns
        -------
        self
        """
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.label_encoder_ = le
        self.n_classes_ = len(le.classes_)

        n_train = len(X)
        has_test = X_test is not None
        n_test = len(X_test) if has_test else 0

        oof_probas: dict[str, np.ndarray] = {
            name: np.zeros((n_train, self.n_classes_)) for name, _ in self.base_models
        }
        if has_test:
            test_probas: dict[str, np.ndarray] = {
                name: np.zeros((n_test, self.n_classes_)) for name, _ in self.base_models
            }

        fold_iter = list(enumerate(cv.split(X, y_enc)))
        for fold, (train_idx, val_idx) in tqdm(fold_iter, desc="CV fold", unit="fold"):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

            fold_models: dict[str, object] = {}
            for name, model in self.base_models:
                m = clone(model)
                kwargs = (model_fit_kwargs or {}).get(name, {})
                m.fit(X_tr, y_tr, **kwargs)
                fold_models[name] = m
                oof_probas[name][val_idx] = m.predict_proba(X_val)
                if has_test:
                    test_probas[name] += m.predict_proba(X_test) / cv.get_n_splits()

            self.fold_models_.append(fold_models)

            fold_preds_enc = np.argmax(
                sum(oof_probas[n][val_idx] for n, _ in self.base_models), axis=1
            )
            score = balanced_accuracy_score(y_val, fold_preds_enc)
            self.valid_scores_.append(score)

        oof_meta = np.hstack([oof_probas[n] for n, _ in self.base_models])
        self.oof_meta_ = oof_meta
        self.y_enc_ = y_enc
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof_meta, y_enc)

        oof_final_enc = self.meta_model_.predict(oof_meta)
        self.overall_oof_score_ = balanced_accuracy_score(y_enc, oof_final_enc)

        if has_test:
            self.test_meta_ = np.hstack([test_probas[n] for n, _ in self.base_models])
        else:
            self.test_meta_ = None

        return self

    def tune_thresholds(self) -> float:
        """Tune per-class score multipliers on OOF meta-probabilities.

        Optimises balanced accuracy by scaling each class's meta-probability
        column-group by a multiplier, then taking argmax.  Uses Nelder-Mead
        simplex search.  Stores the multipliers in ``self.thresholds_`` and
        updates ``self.overall_oof_score_``.

        Returns
        -------
        The tuned balanced accuracy score on OOF.
        """
        from scipy.optimize import minimize

        n_models = len(self.base_models)
        n_classes = self.n_classes_
        oof = self.oof_meta_
        y_enc = self.y_enc_

        reshaped = oof.reshape(oof.shape[0], n_models, n_classes)
        avg_proba = reshaped.mean(axis=1)

        def neg_balanced_acc(multipliers):
            scores = avg_proba * multipliers[np.newaxis, :]
            preds = np.argmax(scores, axis=1)
            return -balanced_accuracy_score(y_enc, preds)

        x0 = np.ones(n_classes)
        result = minimize(neg_balanced_acc, x0, method="Nelder-Mead")
        self.thresholds_ = result.x
        self.overall_oof_score_ = -result.fun
        return -result.fun

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for new data.

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        Array of shape ``(n_samples, n_classes)`` with class probabilities.
        """
        n = len(X)
        n_folds = len(self.fold_models_)

        test_probas: dict[str, np.ndarray] = {
            name: np.zeros((n, self.n_classes_)) for name, _ in self.base_models
        }
        for fold_models in self.fold_models_:
            for name, _ in self.base_models:
                test_probas[name] += fold_models[name].predict_proba(X) / n_folds

        test_meta = np.hstack([test_probas[n] for n, _ in self.base_models])
        return self.meta_model_.predict_proba(test_meta)

    def predict_proba_base_avg(self, X: pd.DataFrame) -> np.ndarray:
        """Average of base-model probabilities across folds, bypassing meta-model.

        Returns
        -------
        Array of shape ``(n_samples, n_classes)`` — simple average across
        all fold × base-model probability estimates.
        """
        n = len(X)
        n_folds = len(self.fold_models_)
        probas = np.zeros((n, self.n_classes_))
        for fold_models in self.fold_models_:
            for name, _ in self.base_models:
                probas += fold_models[name].predict_proba(X) / n_folds
        return probas / len(self.base_models)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for new data.

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        1-D array of string class labels.
        """
        n = len(X)
        n_folds = len(self.fold_models_)

        test_probas: dict[str, np.ndarray] = {
            name: np.zeros((n, self.n_classes_)) for name, _ in self.base_models
        }
        for fold_models in self.fold_models_:
            for name, _ in self.base_models:
                test_probas[name] += fold_models[name].predict_proba(X) / n_folds

        test_meta = np.hstack([test_probas[n] for n, _ in self.base_models])
        preds_enc = self.meta_model_.predict(test_meta)
        return self.label_encoder_.inverse_transform(preds_enc)

    def predict_with_thresholds(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using tuned per-class thresholds instead of the meta-model.

        Averages base-model probabilities across folds and models, applies
        ``self.thresholds_`` multipliers, and takes argmax.  Must be called
        after ``tune_thresholds()``.

        Returns
        -------
        1-D array of string class labels.
        """
        n = len(X)
        n_folds = len(self.fold_models_)
        n_classes = self.n_classes_

        test_probas: dict[str, np.ndarray] = {
            name: np.zeros((n, n_classes)) for name, _ in self.base_models
        }
        for fold_models in self.fold_models_:
            for name, _ in self.base_models:
                test_probas[name] += fold_models[name].predict_proba(X) / n_folds

        avg_proba = np.hstack([test_probas[n] for n, _ in self.base_models])
        avg_proba = avg_proba.reshape(n, len(self.base_models), n_classes).mean(axis=1)
        scores = avg_proba * self.thresholds_[np.newaxis, :]
        preds_enc = np.argmax(scores, axis=1)
        return self.label_encoder_.inverse_transform(preds_enc)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> StackingEnsemble:
        return joblib.load(path)


def train_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train LGBM+XGB+CatBoost with stratified CV, return OOF meta + test preds.

    Returns
    -------
    (oof_meta, test_preds)
        oof_meta: shape ``(n_train, n_models * n_classes)`` – stacked OOF
        probabilities from each base model on each CV fold.
        test_preds: 1-D array of string class labels (simple average across
        base models and folds).
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)
    n_train = len(X_train)
    n_test = len(X_test)

    base_models: list[tuple[str, object]] = [
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
            ),
        ),
        (
            "catboost",
            CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6, random_seed=random_state, verbose=0
            ),
        ),
    ]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_probas: dict[str, np.ndarray] = {
        name: np.zeros((n_train, n_classes)) for name, _ in base_models
    }
    test_probas: dict[str, np.ndarray] = {
        name: np.zeros((n_test, n_classes)) for name, _ in base_models
    }

    for train_idx, val_idx in tqdm(
        cv.split(X_train, y_enc),
        total=n_splits,
        desc="CV fold",
        unit="fold",
    ):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_enc[train_idx]

        for name, model in base_models:
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof_probas[name][val_idx] = m.predict_proba(X_val)
            test_probas[name] += m.predict_proba(X_test) / n_splits

    oof_meta = np.hstack([oof_probas[n] for n, _ in base_models])
    avg_test = sum(test_probas[n] for n, _ in base_models) / len(base_models)
    test_preds = le.inverse_transform(np.argmax(avg_test, axis=1))

    return oof_meta, test_preds


def save_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = "outputs/submissions/submission.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "class": predictions})
    submission.to_csv(out, index=False)
