"""Adaptive XGBoost trainer - regressor or classifier based on target type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from demand_pulse.target_detector import TargetProfile, TargetType


@dataclass(frozen=True)
class TrainingResult:
    """Evaluation metrics and prediction artifacts from model training."""

    task_type: str
    model_name: str
    rmse: float
    mae: float
    r2: float
    accuracy: float
    f1: float
    y_pred: np.ndarray
    y_true: np.ndarray | None
    feature_importance: pd.DataFrame
    hyperparameters: dict[str, Any]
    peak_indices: list[int]
    low_indices: list[int]
    label_encoder: LabelEncoder | None = None
    class_labels: list | None = None

    @property
    def is_classification(self) -> bool:
        return self.task_type in {TargetType.BINARY.value, TargetType.DISCRETE.value}


class ModelTrainer:
    """Train XGBoost regressor or classifier based on detected target type."""

    DEFAULT_REGRESSION_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    DEFAULT_CLASSIFICATION_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }

    def __init__(
        self,
        target_profile: TargetProfile,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        self._target_profile = target_profile
        base = (
            self.DEFAULT_CLASSIFICATION_PARAMS
            if target_profile.target_type != TargetType.CONTINUOUS
            else self.DEFAULT_REGRESSION_PARAMS
        )
        self._hyperparameters = {**base, **(hyperparameters or {})}
        self._label_encoder: LabelEncoder | None = None

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return dict(self._hyperparameters)

    @classmethod
    def default_hyperparameters(cls, target_profile: TargetProfile) -> dict[str, Any]:
        if target_profile.target_type == TargetType.CONTINUOUS:
            return dict(cls.DEFAULT_REGRESSION_PARAMS)
        return dict(cls.DEFAULT_CLASSIFICATION_PARAMS)

    def update_hyperparameters(self, updates: dict[str, Any]) -> None:
        self._hyperparameters.update(updates)

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series | None,
        feature_names: list[str],
    ) -> tuple[Any, TrainingResult]:
        """Fit the appropriate XGBoost model and return evaluation artifacts."""
        y_train_encoded, y_test_encoded, label_encoder, class_labels = self._encode_targets(
            y_train, y_test
        )

        model = self._build_model()
        model.fit(X_train, y_train_encoded)

        raw_pred = model.predict(X_test)
        y_pred = self._decode_predictions(raw_pred, label_encoder)
        y_true = self._decode_predictions(y_test_encoded, label_encoder) if y_test_encoded is not None else None

        metrics = self._compute_metrics(y_true, y_pred, raw_pred, y_test_encoded)

        importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        peak_indices, low_indices = self._identify_extremes(
            raw_pred if self._target_profile.target_type == TargetType.CONTINUOUS else raw_pred.astype(float)
        )

        result = TrainingResult(
            task_type=self._target_profile.target_type.value,
            model_name=self._target_profile.model_name,
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2=metrics["r2"],
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            y_pred=y_pred,
            y_true=y_true,
            feature_importance=importance,
            hyperparameters=dict(self._hyperparameters),
            peak_indices=peak_indices,
            low_indices=low_indices,
            label_encoder=label_encoder,
            class_labels=class_labels,
        )
        return model, result

    def _build_model(self) -> XGBRegressor | XGBClassifier:
        params = dict(self._hyperparameters)
        if self._target_profile.target_type == TargetType.CONTINUOUS:
            params.pop("eval_metric", None)
            return XGBRegressor(**params)

        if self._target_profile.target_type == TargetType.BINARY:
            params["eval_metric"] = "logloss"
            return XGBClassifier(**params)

        params["eval_metric"] = "mlogloss"
        return XGBClassifier(**params)

    def _encode_targets(
        self,
        y_train: pd.Series,
        y_test: pd.Series | None,
    ) -> tuple[np.ndarray, np.ndarray | None, LabelEncoder | None, list | None]:
        if self._target_profile.target_type == TargetType.CONTINUOUS:
            train = y_train.astype(float).to_numpy()
            test = y_test.astype(float).to_numpy() if y_test is not None else None
            return train, test, None, None

        self._label_encoder = LabelEncoder()
        train = self._label_encoder.fit_transform(y_train.astype(str))
        class_labels = self._label_encoder.classes_.tolist()
        test = (
            self._label_encoder.transform(y_test.astype(str))
            if y_test is not None
            else None
        )
        return train, test, self._label_encoder, class_labels

    def _decode_predictions(
        self,
        values: np.ndarray | None,
        label_encoder: LabelEncoder | None,
    ) -> np.ndarray | None:
        if values is None:
            return None
        if label_encoder is None:
            return np.asarray(values, dtype=float)
        return label_encoder.inverse_transform(values.astype(int))

    def _compute_metrics(
        self,
        y_true: np.ndarray | None,
        y_pred: np.ndarray,
        raw_pred: np.ndarray,
        y_test_encoded: np.ndarray | None,
    ) -> dict[str, float]:
        if y_true is None or y_test_encoded is None:
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "accuracy": float("nan"), "f1": float("nan")}

        if self._target_profile.target_type == TargetType.CONTINUOUS:
            y_true_f = np.asarray(y_true, dtype=float)
            y_pred_f = np.asarray(y_pred, dtype=float)
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
                "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
                "r2": float(r2_score(y_true_f, y_pred_f)),
                "accuracy": float("nan"),
                "f1": float("nan"),
            }

        avg = "binary" if self._target_profile.target_type == TargetType.BINARY else "weighted"
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "accuracy": float(accuracy_score(y_test_encoded, raw_pred)),
            "f1": float(f1_score(y_test_encoded, raw_pred, average=avg, zero_division=0)),
        }

    @staticmethod
    def _identify_extremes(
        predictions: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[int], list[int]]:
        if len(predictions) == 0:
            return [], []
        order = np.argsort(predictions)
        low = order[: min(top_k, len(order))].tolist()
        peak = order[-min(top_k, len(order)) :][::-1].tolist()
        return peak, low
