"""Load CSV/Excel data, preprocess features, and produce train/test matrices."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from demand_pulse.column_analyzer import ColumnAnalyzer, ColumnProfile
from demand_pulse.target_detector import TargetDetector, TargetProfile, TargetType


@dataclass(frozen=True)
class DataSplit:
    """Container for prepared train/test matrices."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | None
    y_test: pd.Series | None
    feature_names: list[str]
    feature_columns: list[str]
    preprocessor: ColumnTransformer | None
    dropped_id_columns: list[str]
    has_test_target: bool
    target_profile: TargetProfile


class DataLoader:
    """Ingest tabular demand data and produce model-ready feature matrices."""

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})

    def __init__(
        self,
        train_dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        index_columns: list[str] | None = None,
        test_dataframe: pd.DataFrame | None = None,
    ) -> None:
        self._train_df = train_dataframe.copy()
        self._test_df = test_dataframe.copy() if test_dataframe is not None else None
        self._target_column = target_column
        self._feature_columns = feature_columns
        self._index_columns = index_columns or []

    @staticmethod
    def from_uploaded_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
        """Parse uploaded file bytes into a DataFrame."""
        ext = DataLoader._extension(filename)
        if ext not in DataLoader.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(DataLoader.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported formats: {supported}"
            )

        buffer = io.BytesIO(raw_bytes)
        if ext == ".csv":
            return pd.read_csv(buffer)
        if ext == ".xlsx":
            return pd.read_excel(buffer, engine="openpyxl")
        return pd.read_excel(buffer)

    @staticmethod
    def _extension(filename: str) -> str:
        if "." not in filename:
            return ""
        return filename[filename.rfind(".") :].lower()

    def validate(self) -> None:
        if self._target_column not in self._train_df.columns:
            raise ValueError(
                f"Target column '{self._target_column}' not found in training dataset."
            )
        if self._train_df[self._target_column].isna().all():
            raise ValueError("Target column contains only missing values.")
        missing = [c for c in self._feature_columns if c not in self._train_df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in training data: {missing}")

    def split(self) -> DataSplit:
        """Prepare train/test matrices with preprocessing fitted on training data."""
        self.validate()

        train_df = self._train_df.dropna(subset=[self._target_column]).copy()
        id_cols = [c for c in self._index_columns if c in train_df.columns]
        feature_cols = [
            c for c in self._feature_columns if c in train_df.columns and c not in id_cols
        ]

        y_train_full = train_df[self._target_column]
        X_train_full = train_df[feature_cols]
        target_profile = TargetDetector().analyze(train_df, self._target_column)

        if self._test_df is not None:
            test_df = self._test_df.copy()
            available_test_features = [c for c in feature_cols if c in test_df.columns]
            missing_test_features = [c for c in feature_cols if c not in test_df.columns]
            if missing_test_features:
                feature_cols = available_test_features
            if not feature_cols:
                raise ValueError(
                    "No shared feature columns between training and test datasets."
                )
            X_test_raw = test_df[feature_cols]
            has_test_target = self._target_column in test_df.columns
            y_test_raw = (
                test_df[self._target_column]
                if has_test_target
                else None
            )
            X_train_raw = X_train_full
            y_train_raw = y_train_full
        else:
            stratify = None
            if target_profile.target_type != TargetType.CONTINUOUS:
                stratify = y_train_full.astype(str)
            try:
                X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=self.TEST_SIZE,
                    random_state=self.RANDOM_STATE,
                    stratify=stratify,
                )
            except ValueError:
                X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=self.TEST_SIZE,
                    random_state=self.RANDOM_STATE,
                )
            has_test_target = True

        X_train, X_test, feature_names, preprocessor = self._preprocess(
            X_train_raw, X_test_raw
        )

        return DataSplit(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train_raw.reset_index(drop=True),
            y_test=y_test_raw.reset_index(drop=True) if y_test_raw is not None else None,
            feature_names=feature_names,
            feature_columns=feature_cols,
            preprocessor=preprocessor,
            dropped_id_columns=id_cols,
            has_test_target=has_test_target,
            target_profile=target_profile,
        )

    @staticmethod
    def _preprocess(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], ColumnTransformer | None]:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            )

        if not transformers:
            scaler = StandardScaler()
            feature_names = list(X_train.columns)
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=feature_names,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_names,
            )
            return X_train_scaled, X_test_scaled, feature_names, None

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = DataLoader._build_feature_names(
            preprocessor, numeric_cols, categorical_cols
        )
        X_train_frame = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_frame = pd.DataFrame(X_test_processed, columns=feature_names)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_frame),
            columns=feature_names,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_frame),
            columns=feature_names,
        )
        return X_train_scaled, X_test_scaled, feature_names, preprocessor

    @staticmethod
    def _build_feature_names(
        preprocessor: ColumnTransformer,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> list[str]:
        names: list[str] = list(numeric_cols)
        if not categorical_cols:
            return names

        cat_transformer = preprocessor.named_transformers_.get("cat")
        if cat_transformer is None:
            return names

        if hasattr(cat_transformer, "get_feature_names_out"):
            raw = cat_transformer.get_feature_names_out(categorical_cols).tolist()
            clean: list[str] = []
            for feat in raw:
                parts = feat.split("_", 1)
                if len(parts) == 2 and parts[0].startswith("x"):
                    try:
                        col_idx = int(parts[0][1:])
                        col_name = categorical_cols[col_idx]
                        clean.append(f"{col_name}={parts[1]}")
                    except (ValueError, IndexError):
                        clean.append(feat)
                else:
                    clean.append(feat)
            names.extend(clean)
        else:
            names.extend(categorical_cols)
        return names

    @staticmethod
    def analyze_columns(dataframe: pd.DataFrame) -> ColumnProfile:
        """Convenience wrapper around :class:`ColumnAnalyzer`."""
        return ColumnAnalyzer().analyze(dataframe)
