from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


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

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
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


def save_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = "outputs/submissions/submission.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "class": predictions})
    submission.to_csv(out, index=False)
