"""Qwen 3.5 (via Ollama) - business overview from ML results and data patterns."""

from __future__ import annotations

import json
import re

from demand_pulse.demand_insights import InsightSummary
from demand_pulse.model_trainer import TrainingResult

_BUSINESS_OVERVIEW_PROMPT = """\
You are a Senior Business Consultant presenting to the C-suite of a transportation \
company (such as a bike-sharing operator, ride-hail provider, or urban transit service). \
You have just run a demand forecasting analysis using their operational data and machine \
learning results. Your job is to translate that analysis into a compelling, plain-language \
business case that answers the question every executive asks: "Why should I buy this, and \
what will it do for my bottom line?"

Do NOT use technical terms like R-squared, RMSE, XGBoost, regression, or model accuracy. \
Translate every finding into business language. Speak in terms of riders, revenue, fleet, \
staff, and market opportunity.

---

### WHAT THE DATA TELLS US ###

Target metric being predicted: {target_column}
Total historical records analysed: {total_records}
Forecasting reliability: {reliability_statement}

Conditions when demand is HIGHEST (based on data patterns):
{peak_conditions}

Conditions when demand is LOWEST:
{low_conditions}

Time-of-day and day-of-week patterns:
{temporal_patterns}

How the key environmental and operational factors shift between busy and quiet periods:
{feature_contrast}

---

Write the following five sections. Use plain, confident, boardroom language. \
Be specific - cite the actual numbers and conditions from the data above. \
Do not invent figures that are not in the data. Write in present tense.

#### 1. The Business Problem This Solves
In 2-3 sentences, describe the costly operational problem every transportation operator \
faces: deploying too many vehicles in quiet periods wastes money; too few in peak periods \
loses revenue and customers. Explain that this platform eliminates that guesswork.

#### 2. What the Data Reveals About Your Demand
Describe in plain language the key patterns found: when demand surges, what conditions \
drive it (weather, season, time of day, day of week), and when demand drops. \
Cite the actual values from the data. Explain what this means for a fleet manager or \
operations director making daily decisions.

#### 3. The Financial Opportunity
Based on the demand patterns, estimate the financial upside. Frame it as: \
"During high-demand periods [describe conditions], you can expect significantly more \
users. By having the right number of vehicles or staff ready, you capture that revenue \
instead of turning customers away." \
If the data shows low-demand periods, frame those as cost-saving opportunities: \
"During quiet periods [describe conditions], you can reduce fleet size or staff levels \
without harming service." \
Be concrete about what levers the company can pull.

#### 4. Why This Platform Gives You an Edge
Explain that this is not a one-time report but a live system: the company uploads \
their latest data, the platform analyses demand patterns automatically, and the \
operations team receives actionable guidance within minutes - not weeks. \
Highlight that the system works for any transportation dataset, not just bikes.

#### 5. What We Recommend You Do Next
Give 3 to 5 specific, prioritised actions the company should take based on the \
demand patterns discovered. Frame each as a business action, not a technical task. \
Example: "Increase fleet availability during [specific conditions] - the data shows \
demand is consistently higher under these circumstances."

---

Write with authority and clarity. A non-technical CEO should finish reading this and \
immediately understand both the problem and the opportunity. No JSON. No bullet \
lists of raw numbers. Prose only for sections 1-4, then a numbered action list for section 5.
"""


class LLMAnalyst:
    """Build and deliver the business overview prompt to a local Qwen model."""

    MODEL_NAME: str = "qwen3.5"
    _PREFERRED_BASES: tuple[str, ...] = (
        "qwen3.5",
        "qwen3",
        "qwen2.5",
        "qwen2",
        "llama3.2",
        "llama3.1",
        "llama3",
    )

    def __init__(self, model: str | None = None) -> None:
        self._model = model or self._resolve_model()

    @classmethod
    def _resolve_model(cls) -> str:
        try:
            import ollama

            installed = [m["name"] for m in ollama.list().get("models", [])]
            for base in cls._PREFERRED_BASES:
                for name in installed:
                    if name == base or name.startswith(base + ":"):
                        return name
        except Exception:
            pass
        return cls.MODEL_NAME

    def analyze_demand(
        self,
        training_result: TrainingResult,
        insights: InsightSummary,
        target_column: str,
    ) -> str:
        """Generate a boardroom-ready business overview from ML results and data patterns."""
        profiles = insights.feature_profiles
        metrics = insights.test_ml_results.get("metrics", {})

        reliability_statement = self._reliability_statement(training_result, metrics)
        peak_conditions = self._describe_conditions(
            insights.peak_periods,
            profiles.get("at_peak_predictions", {}),
            "high",
        )
        low_conditions = self._describe_conditions(
            insights.low_periods,
            profiles.get("at_low_predictions", {}),
            "low",
        )
        feature_contrast = self._contrast_features(
            profiles.get("at_peak_predictions", {}),
            profiles.get("at_low_predictions", {}),
        )
        temporal = self._describe_temporal(insights.temporal_summary)

        prompt = _BUSINESS_OVERVIEW_PROMPT.format(
            target_column=target_column,
            total_records=insights.data_summary.get("test_records", "unknown"),
            reliability_statement=reliability_statement,
            peak_conditions=peak_conditions,
            low_conditions=low_conditions,
            temporal_patterns=temporal,
            feature_contrast=feature_contrast,
        )
        return self._clean_text(self._call_ollama(prompt))

    @staticmethod
    def _reliability_statement(
        training_result: TrainingResult,
        metrics: dict,
    ) -> str:
        if metrics.get("status") == "forecast_only":
            return (
                "The model has generated demand forecasts for all test records. "
                "No historical ground-truth was available for this test set, so "
                "reliability is based on training-set performance."
            )
        if training_result.is_classification:
            acc = metrics.get("accuracy", 0)
            return (
                f"The model correctly predicted the demand category "
                f"{acc * 100:.1f}% of the time on unseen data. "
                f"This means roughly {acc * 100:.0f} out of every 100 future periods "
                f"will be categorised correctly."
            )
        r2 = metrics.get("r2", 0)
        mae = metrics.get("mae", 0)
        pct = max(0.0, r2) * 100
        return (
            f"The model explains {pct:.0f}% of the variation in {training_result.task_type} "
            f"demand across unseen data. On average, forecasts are off by approximately "
            f"{mae:.1f} units - giving operations teams a reliable window to plan around."
        )

    @staticmethod
    def _describe_conditions(
        periods: list[dict],
        feature_profile: dict,
        label: str,
    ) -> str:
        if not periods and not feature_profile:
            return f"No clear {label}-demand conditions identified in this dataset."

        lines: list[str] = []
        for key, val in feature_profile.items():
            if isinstance(val, dict) and "mean" in val:
                lines.append(
                    f"  - {key}: average value {val['mean']} "
                    f"(ranging from {val['min']} to {val['max']})"
                )
            elif isinstance(val, dict) and "top_values" in val:
                top = ", ".join(str(k) for k in list(val["top_values"].keys())[:3])
                lines.append(f"  - {key}: most common values are {top}")

        if periods:
            sample = periods[0]
            pred = sample.get("predicted_value") or sample.get("predicted_class", "")
            ts = sample.get("datetime", sample.get("timestamp", ""))
            if ts:
                lines.append(f"  - Example {label}-demand window: {ts}, predicted {pred}")
            else:
                lines.append(f"  - Highest {label}-demand prediction: {pred}")

        return "\n".join(lines) if lines else f"Conditions during {label}-demand periods are consistent with the overall dataset."

    @staticmethod
    def _contrast_features(
        peak_profile: dict,
        low_profile: dict,
    ) -> str:
        if not peak_profile or not low_profile:
            return "Feature contrast data not available."

        shared = set(peak_profile.keys()) & set(low_profile.keys())
        lines: list[str] = []
        for key in list(shared)[:6]:
            p = peak_profile[key]
            q = low_profile[key]
            if isinstance(p, dict) and "mean" in p and isinstance(q, dict) and "mean" in q:
                diff = round(float(p["mean"]) - float(q["mean"]), 2)
                direction = "higher" if diff > 0 else "lower"
                lines.append(
                    f"  - {key} is {abs(diff)} units {direction} during busy periods "
                    f"({p['mean']} vs {q['mean']} in quiet periods)"
                )
        return "\n".join(lines) if lines else "No strong contrasting patterns detected between busy and quiet periods."

    @staticmethod
    def _describe_temporal(temporal_summary: dict) -> str:
        if not temporal_summary:
            return "No time-of-day or day-of-week data available in this dataset."

        lines: list[str] = []
        peak_hours = temporal_summary.get("peak_hours", {})
        if peak_hours:
            hours_str = ", ".join(f"{h}:00" for h in list(peak_hours.keys())[:3])
            lines.append(f"  - Demand tends to peak around {hours_str}")

        peak_days = temporal_summary.get("peak_weekdays", {})
        if peak_days:
            days_str = ", ".join(str(d) for d in list(peak_days.keys())[:3])
            lines.append(f"  - Busiest days of the week: {days_str}")

        return "\n".join(lines) if lines else "Temporal pattern data not available."

    def _call_ollama(self, prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise ConnectionError(
                "The 'ollama' package is not installed. Run: pip install ollama"
            ) from exc

        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            return str(response["message"]["content"])
        except Exception as exc:
            raise ConnectionError(
                f"Ollama request failed. Ensure Ollama is running.\n"
                f"  Pull command: ollama pull {self._model}\n"
                f"  Details: {exc}"
            ) from exc

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text.strip())
