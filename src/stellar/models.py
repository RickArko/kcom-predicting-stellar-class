from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


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
                m.fit(X_tr, y_tr)
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
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof_meta, y_enc)

        oof_final_enc = self.meta_model_.predict(oof_meta)
        self.overall_oof_score_ = balanced_accuracy_score(y_enc, oof_final_enc)

        if has_test:
            self.test_meta_ = np.hstack([test_probas[n] for n, _ in self.base_models])
        else:
            self.test_meta_ = None

        return self

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
