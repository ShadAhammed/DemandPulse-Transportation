"""Detect output column type and recommend the appropriate ML task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class TargetType(str, Enum):
    """Supported prediction task types."""

    BINARY = "binary"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class TargetProfile:
    """Analysis of the selected output column."""

    column: str
    target_type: TargetType
    n_unique: int
    unique_values: list
    model_name: str
    model_description: str
    is_integer_like: bool


class TargetDetector:
    """Infer whether the output column is binary, discrete, or continuous."""

    MAX_DISCRETE_CLASSES: int = 25
    DISCRETE_RATIO: float = 0.05

    def analyze(self, dataframe: pd.DataFrame, column: str) -> TargetProfile:
        """Return task type and recommended model for the output column."""
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        series = dataframe[column].dropna()
        if series.empty:
            raise ValueError(f"Column '{column}' contains only missing values.")

        unique_values = sorted(series.unique(), key=lambda v: (str(type(v)), str(v)))
        n_unique = len(unique_values)
        is_integer_like = self._is_integer_like(series)

        if self._is_categorical_dtype(series):
            target_type = TargetType.BINARY if n_unique == 2 else TargetType.DISCRETE
        elif n_unique == 2:
            target_type = TargetType.BINARY
        elif n_unique <= self._discrete_threshold(len(series)) and is_integer_like:
            target_type = TargetType.DISCRETE
        else:
            target_type = TargetType.CONTINUOUS

        model_name, model_description = self._model_for_type(target_type, n_unique)

        return TargetProfile(
            column=column,
            target_type=target_type,
            n_unique=n_unique,
            unique_values=[self._display_value(v) for v in unique_values[:12]],
            model_name=model_name,
            model_description=model_description,
            is_integer_like=is_integer_like,
        )

    def _discrete_threshold(self, n_rows: int) -> int:
        return min(self.MAX_DISCRETE_CLASSES, max(3, int(n_rows * self.DISCRETE_RATIO)))

    @staticmethod
    def _is_categorical_dtype(series: pd.Series) -> bool:
        return (
            series.dtype == "object"
            or str(series.dtype).startswith("category")
            or str(series.dtype).startswith("string")
        )

    @staticmethod
    def _is_integer_like(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            return False
        return bool(np.allclose(numeric, np.round(numeric)))

    @staticmethod
    def _display_value(value: object) -> object:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    @staticmethod
    def _model_for_type(target_type: TargetType, n_unique: int) -> tuple[str, str]:
        if target_type == TargetType.BINARY:
            return (
                "XGBClassifier",
                "Binary classification - predicts one of two outcome classes",
            )
        if target_type == TargetType.DISCRETE:
            return (
                "XGBClassifier",
                f"Multi-class classification - predicts among {n_unique} discrete categories",
            )
        return (
            "XGBRegressor",
            "Regression - predicts continuous numeric demand values",
        )
