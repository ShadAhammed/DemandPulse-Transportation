"""Infer index, target, and feature columns from arbitrary tabular demand data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    """Suggested column roles for a transportation demand dataset."""

    all_columns: list[str]
    suggested_index_columns: list[str]
    suggested_target_columns: list[str]
    suggested_feature_columns: list[str]
    datetime_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


class ColumnAnalyzer:
    """Analyze schema and recommend column assignments for demand modelling."""

    ID_CARDINALITY_THRESHOLD: float = 0.90
    _ID_NAME_PATTERNS: tuple[str, ...] = (
        "_id",
        "id_",
        "uuid",
        "guid",
        "case_",
        "_key",
        "record_id",
        "row_id",
        "index",
        "instant",
        "dteday",
    )
    _DATETIME_NAME_PATTERNS: tuple[str, ...] = (
        "datetime",
        "timestamp",
        "date_time",
        "time_stamp",
        "date",
        "time",
    )
    _TARGET_NAME_PATTERNS: tuple[str, ...] = (
        "count",
        "demand",
        "rides",
        "trips",
        "volume",
        "usage",
        "passengers",
        "bookings",
        "load",
        "target",
        "y",
    )
    _LEAKAGE_PATTERNS: tuple[str, ...] = (
        "registered",
        "casual",
    )

    def analyze(self, dataframe: pd.DataFrame) -> ColumnProfile:
        """Return column role recommendations for the uploaded dataset."""
        df = dataframe.copy()
        all_columns = list(df.columns)

        datetime_cols = self._detect_datetime_columns(df)
        id_cols = self._detect_id_columns(df)
        id_cols = list(dict.fromkeys(id_cols + datetime_cols))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in all_columns if c not in numeric_cols]

        target_candidates = self._suggest_targets(df, numeric_cols, id_cols)
        leakage_cols = self._detect_leakage_columns(df, target_candidates)
        primary_target = target_candidates[0] if target_candidates else None
        feature_candidates = [
            c
            for c in all_columns
            if c not in id_cols
            and c not in leakage_cols
            and c != primary_target
        ]

        return ColumnProfile(
            all_columns=all_columns,
            suggested_index_columns=id_cols,
            suggested_target_columns=target_candidates,
            suggested_feature_columns=feature_candidates,
            datetime_columns=datetime_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

    def resolve_feature_columns(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        index_columns: list[str],
        selected_features: list[str] | None = None,
        test_dataframe: pd.DataFrame | None = None,
    ) -> list[str]:
        """Return modelling feature columns after excluding index and target."""
        excluded = set(index_columns) | {target_column}
        if selected_features is not None:
            features = [
                c for c in selected_features if c in dataframe.columns and c not in excluded
            ]
        else:
            features = [c for c in dataframe.columns if c not in excluded]

        if test_dataframe is not None:
            test_cols = set(test_dataframe.columns)
            features = [c for c in features if c in test_cols]

        return features

    @classmethod
    def _detect_datetime_columns(cls, df: pd.DataFrame) -> list[str]:
        detected: list[str] = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pat in col_lower for pat in cls._DATETIME_NAME_PATTERNS):
                detected.append(col)
                continue
            if df[col].dtype == "object":
                sample = df[col].dropna().head(20)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    detected.append(col)
        return detected

    @classmethod
    def _detect_id_columns(cls, df: pd.DataFrame) -> list[str]:
        candidates: list[str] = []
        n_rows = len(df)
        for col in df.columns:
            col_lower = col.lower()
            if any(pat in col_lower for pat in cls._ID_NAME_PATTERNS):
                candidates.append(col)
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and n_rows > 0:
                ratio = df[col].nunique() / n_rows
                if ratio >= cls.ID_CARDINALITY_THRESHOLD:
                    candidates.append(col)
        return candidates

    @classmethod
    def _suggest_targets(
        cls,
        df: pd.DataFrame,
        numeric_cols: list[str],
        id_cols: list[str],
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for col in numeric_cols:
            if col in id_cols:
                continue
            col_lower = col.lower()
            series = df[col].dropna()
            if series.empty:
                continue

            score = 0.0
            name_match = any(pat in col_lower for pat in cls._TARGET_NAME_PATTERNS)
            if name_match:
                score += 5.0
            if any(pat in col_lower for pat in cls._LEAKAGE_PATTERNS):
                score += 2.0

            nunique = series.nunique()
            if nunique <= 2 and not name_match:
                continue
            if series.min() >= 0 and series.max() > 10:
                score += 1.0
            if nunique > max(20, len(series) * 0.05):
                score += 1.5

            if col_lower in {"humidity", "temp", "atemp", "windspeed", "season", "weather"}:
                continue

            if score >= 2.0 or name_match:
                scored.append((score, col))

        scored.sort(key=lambda item: item[0], reverse=True)
        ordered = [col for _, col in scored if not cls._detect_leakage_from_names(col)]
        if ordered:
            return ordered[:5]

        fallback = [
            col
            for col in numeric_cols
            if col not in id_cols and df[col].nunique() > 2
        ]
        return fallback[:3]

    @classmethod
    def _detect_leakage_columns(
        cls,
        df: pd.DataFrame,
        target_candidates: list[str],
    ) -> list[str]:
        """Flag columns that are likely components of the demand target."""
        if not target_candidates:
            return []
        primary_target = target_candidates[0]
        if primary_target not in df.columns:
            return []

        leakage: list[str] = []
        target_lower = primary_target.lower()
        for col in df.columns:
            if col == primary_target:
                continue
            col_lower = col.lower()
            if any(pat in col_lower for pat in cls._LEAKAGE_PATTERNS):
                if target_lower in {"count", "total", "demand", "volume"}:
                    leakage.append(col)
        return leakage

    @classmethod
    def _detect_leakage_from_names(cls, column: str) -> bool:
        col_lower = column.lower()
        return any(pat in col_lower for pat in cls._LEAKAGE_PATTERNS)
