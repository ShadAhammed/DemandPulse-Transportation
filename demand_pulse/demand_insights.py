"""Derive insights from raw data, feature values, and ML test-set results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from demand_pulse.model_trainer import TrainingResult


@dataclass(frozen=True)
class InsightSummary:
    """Structured analysis grounded in data, features, and test ML output."""

    task_type: str
    data_summary: dict[str, Any]
    feature_profiles: dict[str, Any]
    test_ml_results: dict[str, Any]
    peak_periods: list[dict[str, Any]]
    low_periods: list[dict[str, Any]]
    temporal_summary: dict[str, Any]
    narrative_bullets: list[str]
    prediction_distribution: dict[str, int]


class DemandInsights:
    """Summarize demand patterns from data, feature values, and test predictions."""

    SAMPLE_ROWS: int = 5

    def summarize(
        self,
        training_result: TrainingResult,
        raw_test_df: pd.DataFrame | None,
        target_column: str,
        index_columns: list[str],
        feature_columns: list[str],
    ) -> InsightSummary:
        """Build analysis from dataset context, feature values, and test ML output."""
        predictions = training_result.y_pred
        context_df = raw_test_df.reset_index(drop=True) if raw_test_df is not None else None

        if training_result.is_classification:
            pred_label = "predicted_class"
        else:
            pred_label = "predicted_value"

        peak_periods = self._build_period_records(
            training_result.peak_indices,
            predictions,
            context_df,
            index_columns,
            feature_columns,
            target_column,
            training_result.y_true,
            pred_label,
        )
        low_periods = self._build_period_records(
            training_result.low_indices,
            predictions,
            context_df,
            index_columns,
            feature_columns,
            target_column,
            training_result.y_true,
            pred_label,
        )

        data_summary = self._data_summary(
            context_df, target_column, predictions, training_result
        )
        feature_profiles = self._feature_profiles(
            context_df,
            feature_columns,
            training_result.peak_indices,
            training_result.low_indices,
        )
        test_ml_results = self._test_ml_results(
            training_result,
            context_df,
            feature_columns,
            target_column,
            index_columns,
            pred_label,
        )
        temporal_summary = self._temporal_patterns(
            context_df, predictions, index_columns, training_result.is_classification
        )
        distribution = (
            pd.Series(predictions).astype(str).value_counts().head(10).to_dict()
        )
        bullets = self._build_bullets(
            training_result,
            data_summary,
            feature_profiles,
            test_ml_results,
            peak_periods,
            temporal_summary,
            target_column,
            distribution,
        )

        return InsightSummary(
            task_type=training_result.task_type,
            data_summary=data_summary,
            feature_profiles=feature_profiles,
            test_ml_results=test_ml_results,
            peak_periods=peak_periods,
            low_periods=low_periods,
            temporal_summary=temporal_summary,
            narrative_bullets=bullets,
            prediction_distribution={str(k): int(v) for k, v in distribution.items()},
        )

    def _data_summary(
        self,
        context_df: pd.DataFrame | None,
        target_column: str,
        predictions: np.ndarray,
        training_result: TrainingResult,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "test_records": len(predictions),
            "target_column": target_column,
            "model": training_result.model_name,
            "task_type": training_result.task_type,
        }

        if context_df is not None and target_column in context_df.columns:
            target_series = context_df[target_column].dropna()
            if training_result.is_classification:
                summary["actual_class_counts"] = (
                    target_series.astype(str).value_counts().head(10).to_dict()
                )
            else:
                summary["actual_target_mean"] = round(float(target_series.mean()), 2)
                summary["actual_target_min"] = round(float(target_series.min()), 2)
                summary["actual_target_max"] = round(float(target_series.max()), 2)

        pred_numeric = pd.to_numeric(pd.Series(predictions), errors="coerce")
        if training_result.is_classification:
            summary["predicted_class_counts"] = (
                pd.Series(predictions).astype(str).value_counts().head(10).to_dict()
            )
        elif pred_numeric.notna().all():
            summary["predicted_mean"] = round(float(pred_numeric.mean()), 2)
            summary["predicted_min"] = round(float(pred_numeric.min()), 2)
            summary["predicted_max"] = round(float(pred_numeric.max()), 2)

        return summary

    def _feature_profiles(
        self,
        context_df: pd.DataFrame | None,
        feature_columns: list[str],
        peak_indices: list[int],
        low_indices: list[int],
    ) -> dict[str, Any]:
        if context_df is None or not feature_columns:
            return {}

        available = [c for c in feature_columns if c in context_df.columns]
        peak_profile = self._aggregate_features(context_df, available, peak_indices)
        low_profile = self._aggregate_features(context_df, available, low_indices)
        overall_profile = self._aggregate_features(
            context_df, available, list(range(min(len(context_df), 500)))
        )

        return {
            "at_peak_predictions": peak_profile,
            "at_low_predictions": low_profile,
            "overall_test_set": overall_profile,
        }

    @staticmethod
    def _aggregate_features(
        df: pd.DataFrame,
        columns: list[str],
        indices: list[int],
    ) -> dict[str, Any]:
        profile: dict[str, Any] = {}
        valid_indices = [i for i in indices if 0 <= i < len(df)]
        if not valid_indices:
            return profile

        subset = df.iloc[valid_indices]
        for col in columns:
            series = subset[col].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                profile[col] = {
                    "mean": round(float(series.mean()), 3),
                    "min": round(float(series.min()), 3),
                    "max": round(float(series.max()), 3),
                }
            else:
                counts = series.astype(str).value_counts().head(3)
                profile[col] = {"top_values": counts.to_dict()}
        return profile

    def _test_ml_results(
        self,
        training_result: TrainingResult,
        context_df: pd.DataFrame | None,
        feature_columns: list[str],
        target_column: str,
        index_columns: list[str],
        pred_label: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metrics": self._metrics_block(training_result),
            "sample_predictions": [],
        }

        if context_df is None:
            return results

        sample_indices = list(dict.fromkeys(
            training_result.peak_indices[: self.SAMPLE_ROWS]
            + training_result.low_indices[: self.SAMPLE_ROWS]
        ))

        for idx in sample_indices:
            if idx >= len(context_df):
                continue
            row = context_df.iloc[idx]
            record: dict[str, Any] = {"index": idx, pred_label: training_result.y_pred[idx]}
            for col in index_columns:
                if col in row.index:
                    record[col] = row[col]
            if training_result.y_true is not None and idx < len(training_result.y_true):
                actual = training_result.y_true[idx]
                record["actual"] = actual
                if training_result.is_classification:
                    record["correct"] = str(actual) == str(training_result.y_pred[idx])
                else:
                    record["error"] = round(
                        float(actual) - float(training_result.y_pred[idx]), 2
                    )
            for col in feature_columns:
                if col in row.index and col not in record:
                    val = row[col]
                    if pd.notna(val):
                        record[col] = val if not isinstance(val, (float, np.floating)) else round(float(val), 3)
            results["sample_predictions"].append(record)

        if not training_result.is_classification and training_result.y_true is not None:
            errors = np.asarray(training_result.y_true, dtype=float) - np.asarray(
                training_result.y_pred, dtype=float
            )
            results["error_summary"] = {
                "mean_error": round(float(np.mean(errors)), 2),
                "mean_abs_error": round(float(np.mean(np.abs(errors))), 2),
                "max_over_prediction": round(float(errors.min()), 2),
                "max_under_prediction": round(float(errors.max()), 2),
            }

        if training_result.is_classification and training_result.y_true is not None:
            correct = sum(
                1
                for i in range(len(training_result.y_true))
                if str(training_result.y_true[i]) == str(training_result.y_pred[i])
            )
            results["correct_predictions"] = correct
            results["incorrect_predictions"] = len(training_result.y_true) - correct

        return results

    @staticmethod
    def _metrics_block(training_result: TrainingResult) -> dict[str, Any]:
        if training_result.y_true is None:
            return {"status": "forecast_only", "predictions": len(training_result.y_pred)}

        if training_result.is_classification:
            return {
                "accuracy": round(training_result.accuracy, 4),
                "f1": round(training_result.f1, 4),
            }
        return {
            "rmse": round(training_result.rmse, 3),
            "mae": round(training_result.mae, 3),
            "r2": round(training_result.r2, 4),
        }

    @staticmethod
    def _build_period_records(
        indices: list[int],
        predictions: np.ndarray,
        context_df: pd.DataFrame | None,
        index_columns: list[str],
        feature_columns: list[str],
        target_column: str,
        y_true: np.ndarray | None,
        pred_label: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx in indices:
            if idx >= len(predictions):
                continue
            value = predictions[idx]
            record: dict[str, Any] = {
                "index": idx,
                pred_label: value
                if not isinstance(value, (float, np.floating))
                else round(float(value), 2),
            }
            if y_true is not None and idx < len(y_true):
                record["actual"] = y_true[idx]
            if context_df is not None and idx < len(context_df):
                for col in index_columns:
                    if col in context_df.columns:
                        record[col] = str(context_df.iloc[idx][col])
                for col in feature_columns:
                    if col in context_df.columns and col != target_column:
                        val = context_df.iloc[idx][col]
                        if pd.notna(val):
                            record[col] = (
                                val
                                if not isinstance(val, (float, np.floating))
                                else round(float(val), 3)
                            )
            records.append(record)
        return records

    @staticmethod
    def _temporal_patterns(
        context_df: pd.DataFrame | None,
        predictions: np.ndarray,
        index_columns: list[str],
        is_classification: bool,
    ) -> dict[str, Any]:
        if context_df is None or context_df.empty:
            return {}

        datetime_col = next(
            (c for c in index_columns if c in context_df.columns),
            None,
        )
        if datetime_col is None:
            for col in context_df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    datetime_col = col
                    break

        if datetime_col is None:
            return {}

        series = pd.to_datetime(context_df[datetime_col], errors="coerce")
        valid = series.notna()
        if valid.sum() < 5:
            return {}

        metric = pd.to_numeric(predictions[valid.to_numpy()], errors="coerce")
        if is_classification and metric.isna().any():
            metric = pd.Series(predictions[valid.to_numpy()]).astype(str)

        frame = pd.DataFrame({"timestamp": series[valid], "value": metric})
        frame["hour"] = frame["timestamp"].dt.hour
        frame["weekday"] = frame["timestamp"].dt.day_name()

        if is_classification:
            hourly = (
                frame.groupby("hour")["value"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .head(5)
                .to_dict()
            )
            daily = (
                frame.groupby("weekday")["value"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .head(5)
                .to_dict()
            )
        else:
            hourly = (
                frame.groupby("hour")["value"]
                .mean()
                .sort_values(ascending=False)
                .head(3)
                .round(2)
                .to_dict()
            )
            daily = (
                frame.groupby("weekday")["value"]
                .mean()
                .sort_values(ascending=False)
                .head(3)
                .round(2)
                .to_dict()
            )
        return {"peak_hours": hourly, "peak_weekdays": daily}

    @staticmethod
    def _build_bullets(
        training_result: TrainingResult,
        data_summary: dict[str, Any],
        feature_profiles: dict[str, Any],
        test_ml_results: dict[str, Any],
        peak_periods: list[dict[str, Any]],
        temporal_summary: dict[str, Any],
        target_column: str,
        distribution: dict[str, int],
    ) -> list[str]:
        """Produce business-language insight bullets for display and LLM context."""
        bullets: list[str] = []
        metrics = test_ml_results.get("metrics", {})
        n = data_summary.get("test_records", 0)

        # Reliability
        if metrics.get("status") == "forecast_only":
            bullets.append(
                f"The platform analysed {n:,} demand records and generated forecasts "
                f"for every time period in the dataset."
            )
        elif training_result.is_classification:
            acc = metrics.get("accuracy", 0)
            bullets.append(
                f"The platform correctly predicted the demand level {acc * 100:.1f}% "
                f"of the time across {n:,} real test periods."
            )
        else:
            r2 = metrics.get("r2", 0)
            mae = metrics.get("mae", 0)
            pred_min = data_summary.get("predicted_min", "?")
            pred_max = data_summary.get("predicted_max", "?")
            bullets.append(
                f"Across {n:,} test periods, the platform explained "
                f"{max(0.0, r2) * 100:.0f}% of demand variation and predicted "
                f"{target_column} values between {pred_min} and {pred_max}, "
                f"with an average error of {mae:.1f} units."
            )

        # Peak conditions
        peak_feats = feature_profiles.get("at_peak_predictions", {})
        low_feats = feature_profiles.get("at_low_predictions", {})
        if peak_feats:
            conditions: list[str] = []
            for feat, val in list(peak_feats.items())[:3]:
                if isinstance(val, dict) and "mean" in val:
                    conditions.append(f"{feat} around {val['mean']}")
                elif isinstance(val, dict) and "top_values" in val:
                    top = list(val["top_values"].keys())[0]
                    conditions.append(f"{feat} = {top}")
            if conditions:
                bullets.append(
                    f"Demand is highest when: {', '.join(conditions)}. "
                    f"These are the conditions to prepare for with extra capacity."
                )

        # Low conditions
        if low_feats:
            quiet: list[str] = []
            for feat, val in list(low_feats.items())[:3]:
                if isinstance(val, dict) and "mean" in val:
                    quiet.append(f"{feat} around {val['mean']}")
                elif isinstance(val, dict) and "top_values" in val:
                    top = list(val["top_values"].keys())[0]
                    quiet.append(f"{feat} = {top}")
            if quiet:
                bullets.append(
                    f"Demand is lowest when: {', '.join(quiet)}. "
                    f"These periods are cost-saving opportunities - fewer vehicles or staff needed."
                )

        # Temporal
        if temporal_summary.get("peak_hours"):
            hours = ", ".join(f"{h}:00" for h in list(temporal_summary["peak_hours"].keys())[:3])
            bullets.append(
                f"The busiest hours of the day are {hours}. "
                f"Scheduling more resources during these windows directly protects revenue."
            )
        if temporal_summary.get("peak_weekdays"):
            days = ", ".join(str(d) for d in list(temporal_summary["peak_weekdays"].keys())[:3])
            bullets.append(f"Busiest days of the week: {days}.")

        # Contrast
        if peak_feats and low_feats:
            shared = set(peak_feats.keys()) & set(low_feats.keys())
            for feat in list(shared)[:2]:
                p = peak_feats[feat]
                q = low_feats[feat]
                if isinstance(p, dict) and "mean" in p and isinstance(q, dict) and "mean" in q:
                    diff = round(abs(float(p["mean"]) - float(q["mean"])), 2)
                    higher = p["mean"] if float(p["mean"]) > float(q["mean"]) else q["mean"]
                    direction = "higher" if float(p["mean"]) > float(q["mean"]) else "lower"
                    bullets.append(
                        f"When {target_column} demand peaks, {feat} is consistently "
                        f"{diff} units {direction} than during quiet periods "
                        f"({higher} vs {q['mean'] if direction == 'higher' else p['mean']}). "
                        f"This pattern is a reliable advance signal for the operations team."
                    )

        return bullets
