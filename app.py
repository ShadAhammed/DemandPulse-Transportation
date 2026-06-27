"""
DemandPulse - Transportation Demand Intelligence Dashboard.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

# ── Professional theme: Midnight Slate & Amber ─────────────────────────────
THEME = {
    "bg": "#0F172A",
    "surface": "#1E293B",
    "surface_alt": "#273449",
    "border": "#475569",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "primary": "#F59E0B",
    "accent": "#38BDF8",
    "success": "#34D399",
    "danger": "#F87171",
}

matplotlib.rcParams.update(
    {
        "figure.facecolor": THEME["surface"],
        "axes.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["border"],
        "axes.labelcolor": THEME["muted"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": THEME["text"],
        "axes.grid": True,
        "grid.color": THEME["surface_alt"],
        "grid.linewidth": 0.5,
        "xtick.color": THEME["muted"],
        "xtick.labelsize": 8,
        "ytick.color": THEME["muted"],
        "ytick.labelsize": 8,
        "text.color": THEME["text"],
        "legend.fontsize": 8,
        "legend.framealpha": 0.4,
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.family": "sans-serif",
        "font.size": 9,
    }
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demand_pulse.column_analyzer import ColumnAnalyzer
from demand_pulse.data_loader import DataLoader
from demand_pulse.demand_insights import DemandInsights
from demand_pulse.llm_analyst import LLMAnalyst
from demand_pulse.model_trainer import ModelTrainer
from demand_pulse.session import (
    STAGE_DONE,
    STAGE_IDLE,
    STAGE_INSIGHTS,
    STAGE_LLM,
    STAGE_PREP,
    STAGE_TRAIN,
    SessionKeys,
    init_session_state,
    reset_pipeline_artifacts,
)
from demand_pulse.target_detector import TargetDetector, TargetType

st.set_page_config(
    page_title="DemandPulse | Demand Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_session_state()

T = THEME
st.markdown(
    f"""
<style>
html,body,[data-testid="stAppViewContainer"]{{background:{T["bg"]}!important;color:{T["text"]}!important}}
[data-testid="stApp"]{{background:{T["bg"]}!important}}
[data-testid="stHeader"]{{background:rgba(15,23,42,.94)!important;border-bottom:1px solid {T["border"]};backdrop-filter:blur(8px)}}
.main .block-container{{padding:1.4rem 2rem 2rem;max-width:1440px}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,{T["surface"]} 0%,{T["bg"]} 100%)!important;border-right:1px solid {T["border"]}}}
[data-testid="stSidebar"] *{{color:{T["muted"]}!important}}
[data-testid="stSidebar"] h1{{font-size:1.1rem!important;font-weight:700!important;color:{T["text"]}!important}}
.stTabs [data-baseweb="tab-list"]{{background:{T["surface"]}!important;border-bottom:1px solid {T["border"]}!important;gap:0!important;padding:0!important}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;color:{T["muted"]}!important;font-size:.875rem!important;font-weight:500!important;padding:.6rem 1.3rem!important}}
.stTabs [aria-selected="true"]{{color:{T["text"]}!important;border-bottom:2px solid {T["primary"]}!important}}
h1{{font-size:1.45rem!important;font-weight:700!important;color:{T["text"]}!important}}
h2{{font-size:1.1rem!important;font-weight:600!important;color:{T["text"]}!important}}
.page-title{{font-size:1.4rem;font-weight:700;color:{T["text"]};letter-spacing:-.01em;margin-bottom:.15rem}}
.page-sub{{font-size:.8rem;color:{T["muted"]};margin-bottom:1.1rem}}
[data-testid="stMetric"]{{background:{T["surface"]}!important;border:1px solid {T["border"]}!important;border-radius:10px!important;padding:.8rem 1rem!important}}
[data-testid="stMetricLabel"]{{color:{T["muted"]}!important;font-size:.75rem!important}}
[data-testid="stMetricValue"]{{color:{T["text"]}!important;font-size:1.3rem!important;font-weight:600!important}}
[data-testid="stVerticalBlockBorderWrapper"]{{background:{T["surface"]}!important;border:1px solid {T["border"]}!important;border-radius:10px!important}}
.stButton>button[kind="primary"]{{background:linear-gradient(135deg,{T["primary"]} 0%,#D97706 100%)!important;border:none!important;color:#0F172A!important;font-weight:700!important;border-radius:8px!important}}
.stButton>button[kind="secondary"]{{background:{T["surface_alt"]}!important;border:1px solid {T["border"]}!important;color:{T["text"]}!important;border-radius:8px!important}}
.stSelectbox [data-baseweb="select"]>div,.stTextInput>div>div>input,.stNumberInput>div>div>input{{background:{T["surface_alt"]}!important;border:1px solid {T["border"]}!important;border-radius:8px!important;color:{T["text"]}!important}}
.stFileUploader{{background:{T["surface"]}!important;border:1px dashed {T["border"]}!important;border-radius:10px!important;padding:.8rem!important}}
[data-testid="stDataFrame"]{{background:{T["surface"]}!important;border:1px solid {T["border"]}!important;border-radius:10px!important}}
.report-label{{font-size:.7rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:{T["accent"]};margin-bottom:.35rem}}
.pipeline-step{{display:flex;align-items:center;gap:.5rem;padding:.35rem 0;font-size:.82rem;color:{T["muted"]}}}
.pipeline-step.active{{color:{T["accent"]};font-weight:600}}
.pipeline-step.done{{color:{T["success"]}}}
.type-badge{{display:inline-block;padding:.25rem .75rem;border-radius:999px;font-size:.75rem;font-weight:600;margin-right:.5rem}}
.type-binary{{background:rgba(56,189,248,.15);color:{T["accent"]};border:1px solid rgba(56,189,248,.35)}}
.type-discrete{{background:rgba(245,158,11,.15);color:{T["primary"]};border:1px solid rgba(245,158,11,.35)}}
.type-continuous{{background:rgba(52,211,153,.15);color:{T["success"]};border:1px solid rgba(52,211,153,.35)}}
.model-card{{background:{T["surface_alt"]};border:1px solid {T["border"]};border-left:4px solid {T["primary"]};border-radius:10px;padding:1rem 1.2rem;margin:.75rem 0}}
</style>
""",
    unsafe_allow_html=True,
)


def _sk(key: SessionKeys) -> Any:
    return st.session_state.get(key)


def _data_ready() -> bool:
    return bool(_sk(SessionKeys.DATA_READY))


def _trained() -> bool:
    return bool(_sk(SessionKeys.TRAINED))


def _type_badge(target_type: TargetType) -> str:
    labels = {
        TargetType.BINARY: ("Binary", "type-binary"),
        TargetType.DISCRETE: ("Discrete", "type-discrete"),
        TargetType.CONTINUOUS: ("Continuous", "type-continuous"),
    }
    label, css = labels[target_type]
    return f'<span class="type-badge {css}">{label}</span>'


def _render_llm_report(markdown_text: str) -> None:
    st.markdown(f'<div class="report-label">Executive Demand Briefing</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(markdown_text or "_No narrative returned._")


def _prepare_data(
    target_col: str,
    index_cols: list[str],
    feature_cols: list[str],
) -> bool:
    if not feature_cols:
        st.error("Select at least one feature column before preparing the pipeline.")
        return False
    try:
        loader = DataLoader(
            train_dataframe=_sk(SessionKeys.DF_TRAIN),
            target_column=target_col,
            feature_columns=feature_cols,
            index_columns=index_cols,
            test_dataframe=_sk(SessionKeys.DF_TEST),
        )
        split = loader.split()
        st.session_state[SessionKeys.X_TRAIN] = split.X_train
        st.session_state[SessionKeys.X_TEST] = split.X_test
        st.session_state[SessionKeys.Y_TRAIN] = split.y_train
        st.session_state[SessionKeys.Y_TEST] = split.y_test
        st.session_state[SessionKeys.FEATURE_NAMES] = split.feature_names
        st.session_state[SessionKeys.FEATURE_COLS] = split.feature_columns
        st.session_state[SessionKeys.TARGET_PROFILE] = split.target_profile
        st.session_state[SessionKeys.DATA_READY] = True
        st.session_state[SessionKeys.HAS_TEST_TARGET] = split.has_test_target
        st.session_state[SessionKeys.DROPPED_ID_COLS] = split.dropped_id_columns
        st.session_state[SessionKeys.MODEL_PARAMS] = ModelTrainer.default_hyperparameters(
            split.target_profile
        )
        st.session_state[SessionKeys.PIPELINE_STAGE] = STAGE_PREP
        return True
    except Exception as exc:
        st.error(f"Data preparation failed: {exc}")
        return False


def _run_train() -> None:
    profile = _sk(SessionKeys.TARGET_PROFILE)
    if profile is None:
        raise ValueError("Target profile missing. Prepare the pipeline first.")
    trainer = ModelTrainer(profile, hyperparameters=_sk(SessionKeys.MODEL_PARAMS))
    model, result = trainer.train_and_evaluate(
        _sk(SessionKeys.X_TRAIN),
        _sk(SessionKeys.Y_TRAIN),
        _sk(SessionKeys.X_TEST),
        _sk(SessionKeys.Y_TEST),
        _sk(SessionKeys.FEATURE_NAMES),
    )
    st.session_state[SessionKeys.MODEL] = model
    st.session_state[SessionKeys.TRAINING_RESULT] = result
    st.session_state[SessionKeys.TRAINED] = True
    st.session_state[SessionKeys.PIPELINE_STAGE] = STAGE_TRAIN
    st.session_state[SessionKeys.TEST_PREDICTIONS] = result.y_pred


def _run_insights() -> None:
    analyzer = DemandInsights()
    raw_test = _sk(SessionKeys.DF_TEST)
    if raw_test is None and _sk(SessionKeys.DF_TRAIN) is not None:
        raw_test = _sk(SessionKeys.DF_TRAIN).tail(len(_sk(SessionKeys.TEST_PREDICTIONS)))
    insights = analyzer.summarize(
        _sk(SessionKeys.TRAINING_RESULT),
        raw_test,
        _sk(SessionKeys.TARGET_COL) or "demand",
        _sk(SessionKeys.INDEX_COLS) or [],
        _sk(SessionKeys.FEATURE_COLS) or [],
    )
    st.session_state[SessionKeys.INSIGHTS] = insights
    st.session_state[SessionKeys.PIPELINE_STAGE] = STAGE_INSIGHTS


def _run_llm() -> None:
    analyst = LLMAnalyst()
    report = analyst.analyze_demand(
        _sk(SessionKeys.TRAINING_RESULT),
        _sk(SessionKeys.INSIGHTS),
        _sk(SessionKeys.TARGET_COL) or "demand",
    )
    st.session_state[SessionKeys.LLM_REPORT] = report
    st.session_state[SessionKeys.LLM_ANALYZED] = True
    st.session_state[SessionKeys.PIPELINE_STAGE] = STAGE_DONE


def _plot_predictions(result: Any) -> None:
    if result.is_classification:
        _plot_class_distribution(result)
        if result.y_true is not None:
            _plot_confusion_matrix(result)
        return

    y_pred = result.y_pred
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    fig.patch.set_facecolor(THEME["surface"])
    ax.set_facecolor(THEME["bg"])
    ax.plot(y_pred, color=THEME["accent"], linewidth=1.2, label="Predicted")
    if result.y_true is not None:
        ax.plot(result.y_true, color=THEME["success"], linewidth=1.0, alpha=0.75, label="Actual")
    ax.set_title("Demand Forecast on Test Set", fontsize=9, color=THEME["text"])
    ax.set_xlabel("Record index", fontsize=8, color=THEME["muted"])
    ax.set_ylabel("Demand", fontsize=8, color=THEME["muted"])
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_class_distribution(result: Any) -> None:
    counts = pd.Series(result.y_pred).astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    fig.patch.set_facecolor(THEME["surface"])
    ax.set_facecolor(THEME["bg"])
    ax.bar(counts.index.astype(str), counts.values, color=THEME["primary"], edgecolor="none")
    ax.set_title("Predicted Class Distribution", fontsize=9, color=THEME["text"])
    ax.set_xlabel("Predicted class", fontsize=8, color=THEME["muted"])
    ax.set_ylabel("Count", fontsize=8, color=THEME["muted"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _plot_confusion_matrix(result: Any) -> None:
    labels = result.class_labels or sorted(set(map(str, result.y_true)) | set(map(str, result.y_pred)))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    fig.patch.set_facecolor(THEME["surface"])
    ConfusionMatrixDisplay.from_predictions(
        result.y_true,
        result.y_pred,
        labels=labels,
        ax=ax,
        colorbar=False,
    )
    ax.set_facecolor(THEME["bg"])
    ax.set_title("Confusion Matrix", fontsize=9, color=THEME["text"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_hyperparam_editor() -> dict[str, Any]:
    profile = _sk(SessionKeys.TARGET_PROFILE)
    current = dict(
        _sk(SessionKeys.MODEL_PARAMS)
        or (ModelTrainer.default_hyperparameters(profile) if profile else {})
    )
    updated: dict[str, Any] = {}
    cols = st.columns(3)
    keys = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"]
    for i, key in enumerate(keys):
        val = current.get(key, 0)
        with cols[i % 3]:
            if key in {"n_estimators", "max_depth", "min_child_weight"}:
                updated[key] = st.number_input(key, min_value=1, max_value=2000, value=int(val), key=f"hp_{key}")
            else:
                updated[key] = st.number_input(
                    key, min_value=0.0001, max_value=1.0, value=float(val), step=0.01, format="%.4f", key=f"hp_{key}"
                )
    updated["random_state"] = current.get("random_state", 42)
    updated["n_jobs"] = current.get("n_jobs", -1)
    if profile and profile.target_type != TargetType.CONTINUOUS:
        updated["eval_metric"] = current.get("eval_metric", "logloss")
    return updated


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ DemandPulse")
    st.caption("Demand Intelligence · Not Auto-XAI")
    st.divider()

    stage = _sk(SessionKeys.PIPELINE_STAGE) or STAGE_IDLE

    def _step(label: str, active: bool, done: bool) -> None:
        cls = "done" if done else ("active" if active else "")
        icon = "✓" if done else ("●" if active else "○")
        st.markdown(f'<div class="pipeline-step {cls}">{icon} {label}</div>', unsafe_allow_html=True)

    _step("Data & Architecture", active=not _data_ready(), done=_data_ready())
    _step("Model", active=stage == STAGE_TRAIN, done=_trained())
    _step("Overview", active=stage in {STAGE_INSIGHTS, STAGE_LLM, STAGE_DONE}, done=bool(_sk(SessionKeys.LLM_ANALYZED)))
    st.divider()

    if _data_ready():
        st.success("Pipeline ready - open the **Model** tab to train.")
    else:
        st.info("Complete **Data & Architecture** and click **Prepare pipeline**.")

    profile = _sk(SessionKeys.TARGET_PROFILE)
    if profile and _data_ready():
        st.markdown(
            f'<div class="model-card">'
            f'<strong>Auto-selected model</strong><br/>'
            f'{profile.model_name}<br/>'
            f'<span style="color:{THEME["muted"]};font-size:.78rem">{profile.model_description}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if _trained():
        tr = _sk(SessionKeys.TRAINING_RESULT)
        if tr.is_classification and tr.y_true is not None:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{tr.accuracy:.3f}")
            c2.metric("F1", f"{tr.f1:.3f}")
        elif tr.y_true is not None:
            c1, c2 = st.columns(2)
            c1.metric("R²", f"{tr.r2:.3f}")
            c2.metric("RMSE", f"{tr.rmse:.1f}")
        else:
            st.metric("Predictions", len(tr.y_pred))

st.markdown(
    f'<div class="page-title">⬡ DemandPulse · Transportation Demand Intelligence</div>'
    f'<div class="page-sub">'
    f'Data &amp; Architecture · Model · Overview'
    f'</div>',
    unsafe_allow_html=True,
)

tab_data, tab_model, tab_overview = st.tabs(
    [" Data & Architecture ", " Model ", " Overview "]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 - Data & Architecture
# ══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### Dataset & Column Architecture")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_train = st.file_uploader("Training dataset (CSV or Excel)", type=["csv", "xlsx", "xls"], key="train_up")
    with col_up2:
        uploaded_test = st.file_uploader(
            "Optional test dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="test_up",
        )

    sample_path = os.path.join(_ROOT, "data", "train.csv")
    if uploaded_train is None and os.path.exists(sample_path) and _sk(SessionKeys.DF_TRAIN) is None:
        st.session_state[SessionKeys.DF_TRAIN] = pd.read_csv(sample_path)
        test_sample = os.path.join(_ROOT, "data", "test.csv")
        if os.path.exists(test_sample):
            st.session_state[SessionKeys.DF_TEST] = pd.read_csv(test_sample)

    if uploaded_train is not None:
        try:
            df_new = DataLoader.from_uploaded_file(uploaded_train.getvalue(), uploaded_train.name)
            if _sk(SessionKeys.DF_TRAIN) is None or not df_new.equals(_sk(SessionKeys.DF_TRAIN)):
                st.session_state[SessionKeys.DF_TRAIN] = df_new
                reset_pipeline_artifacts()
        except Exception as exc:
            st.error(f"Could not read training file: {exc}")

    if uploaded_test is not None:
        try:
            df_test_new = DataLoader.from_uploaded_file(uploaded_test.getvalue(), uploaded_test.name)
            prev_test = _sk(SessionKeys.DF_TEST)
            if prev_test is None or not df_test_new.equals(prev_test):
                st.session_state[SessionKeys.DF_TEST] = df_test_new
                reset_pipeline_artifacts()
        except Exception as exc:
            st.error(f"Could not read test file: {exc}")

    df_train = _sk(SessionKeys.DF_TRAIN)
    if df_train is not None:
        profile = ColumnAnalyzer().analyze(df_train)
        st.session_state[SessionKeys.COLUMN_PROFILE] = profile

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df_train):,}")
        c2.metric("Columns", len(df_train.columns))
        c3.metric("Numeric", len(profile.numeric_columns))
        c4.metric("Categorical", len(profile.categorical_columns))

        st.dataframe(df_train.head(12), use_container_width=True, height=280)

        st.markdown("#### Column roles")

        target_default = (
            profile.suggested_target_columns[0]
            if profile.suggested_target_columns
            else df_train.columns[-1]
        )
        target_col = st.selectbox(
            "Output column (target)",
            options=list(df_train.columns),
            index=list(df_train.columns).index(target_default)
            if target_default in df_train.columns
            else 0,
            key="target_select",
        )

        try:
            target_profile = TargetDetector().analyze(df_train, target_col)
            st.markdown(
                f'{_type_badge(target_profile.target_type)}'
                f'<span style="color:{THEME["muted"]};font-size:.82rem">'
                f'{target_profile.n_unique} unique values · '
                f'Auto-selects <strong style="color:{THEME["text"]}">{target_profile.model_name}</strong>'
                f"</span>",
                unsafe_allow_html=True,
            )
            st.caption(target_profile.model_description)
            if target_profile.unique_values:
                st.caption(f"Sample values: {', '.join(map(str, target_profile.unique_values[:8]))}")
        except ValueError as exc:
            st.warning(str(exc))

        index_cols = st.multiselect(
            "Index / identifier columns (excluded from features)",
            options=list(df_train.columns),
            default=[c for c in profile.suggested_index_columns if c in df_train.columns],
            key="index_select",
        )

        available_features = [
            c for c in df_train.columns if c not in set(index_cols) | {target_col}
        ]
        df_test = _sk(SessionKeys.DF_TEST)
        if df_test is not None:
            available_features = [c for c in available_features if c in df_test.columns]

        profile_features = [
            c for c in profile.suggested_feature_columns if c in available_features
        ]
        default_features = profile_features or available_features

        feature_cols = st.multiselect(
            "Feature columns for modelling",
            options=available_features,
            default=default_features,
            key="feature_select",
        )

        if df_test is not None:
            dropped = [
                c
                for c in df_train.columns
                if c not in set(index_cols) | {target_col} and c not in df_test.columns
            ]
            if dropped:
                st.caption(
                    "Features not in test file (excluded from default selection): "
                    + ", ".join(dropped)
                )

        if feature_cols:
            st.markdown("**Features that will be used for training:**")
            st.code(", ".join(feature_cols), language=None)

        if st.button("Prepare pipeline", type="primary", key="prep_btn"):
            st.session_state[SessionKeys.TARGET_COL] = target_col
            st.session_state[SessionKeys.INDEX_COLS] = index_cols
            st.session_state[SessionKeys.FEATURE_COLS] = feature_cols
            if _prepare_data(target_col, index_cols, feature_cols):
                tp = _sk(SessionKeys.TARGET_PROFILE)
                st.success(
                    f"Ready · {tp.model_name} · {len(feature_cols)} raw features → "
                    f"{len(_sk(SessionKeys.FEATURE_NAMES))} model features · "
                    f"Go to the **Model** tab to train."
                )
                st.rerun()

        if _data_ready():
            st.success("✓ Pipeline prepared. Switch to the **Model** tab.")
    else:
        st.info("Upload a training CSV/Excel file or use the bundled bike-sharing sample in `data/train.csv`.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 - Model
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### Model Training")

    if not _data_ready():
        st.warning(
            "Pipeline not prepared yet. On **Data & Architecture**, select your columns "
            "and click **Prepare pipeline**."
        )
    else:
        tp = _sk(SessionKeys.TARGET_PROFILE)
        if tp:
            st.info(f"Training **{tp.model_name}** for **{tp.target_type.value}** output (`{tp.column}`)")

        hp = _render_hyperparam_editor()
        st.session_state[SessionKeys.MODEL_PARAMS] = hp

        run_col, _ = st.columns([1, 2])
        with run_col:
            if st.button("Train model", type="primary", key="train_btn"):
                with st.spinner("Training …"):
                    _run_train()
                st.success("Model trained.")
                st.rerun()

            if st.button("Run Full Pipeline", type="secondary", key="full_pipe"):
                progress = st.progress(0.0, text="Training …")
                _run_train()
                progress.progress(0.35, text="Analysing test data …")
                _run_insights()
                progress.progress(0.65, text="Querying Qwen 3 …")
                try:
                    _run_llm()
                    progress.progress(1.0, text="Complete")
                    st.success("Full pipeline complete.")
                except ConnectionError as exc:
                    st.warning(f"LLM step skipped: {exc}")
                st.rerun()

        if _trained():
            result = _sk(SessionKeys.TRAINING_RESULT)
            if result.is_classification and result.y_true is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{result.accuracy:.4f}")
                m2.metric("F1", f"{result.f1:.4f}")
                m3.metric("Test records", len(result.y_pred))
            elif result.y_true is not None:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²", f"{result.r2:.4f}")
                m2.metric("RMSE", f"{result.rmse:.2f}")
                m3.metric("MAE", f"{result.mae:.2f}")
                m4.metric("Test records", len(result.y_pred))
            else:
                st.info("Test set has no target column - showing predictions only.")
                st.metric("Predictions generated", len(result.y_pred))

            _plot_predictions(result)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 - Overview
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Business Demand Overview")

    if not _trained():
        st.warning("Train the model on the **Model** tab first.")
    else:
        if _sk(SessionKeys.INSIGHTS) is None:
            _run_insights()

        insights = _sk(SessionKeys.INSIGHTS)

        if insights is not None:
            # --- Demand signals at a glance -----------------------------------
            st.markdown("#### What the data is telling you")
            for bullet in insights.narrative_bullets:
                st.markdown(
                    f'<div style="padding:0.4rem 0.8rem;margin:0.3rem 0;'
                    f'border-left:3px solid #f59e0b;background:#1e293b;'
                    f'border-radius:4px;font-size:0.97rem;">{bullet}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

        # --- Executive briefing (LLM) ----------------------------------------
        st.markdown("### Executive Briefing")

        col_gen, col_ref = st.columns([2, 1])
        with col_gen:
            if st.button(
                "Generate business overview",
                type="primary",
                key="llm_btn",
                use_container_width=True,
            ):
                with st.spinner("Analysing demand patterns and preparing your briefing …"):
                    try:
                        _run_llm()
                        st.rerun()
                    except ConnectionError as exc:
                        st.error(str(exc))
        with col_ref:
            if st.button(
                "Refresh signals",
                type="secondary",
                key="overview_refresh_btn",
                use_container_width=True,
            ):
                _run_insights()
                st.rerun()

        if _sk(SessionKeys.LLM_REPORT):
            _render_llm_report(_sk(SessionKeys.LLM_REPORT))

        # --- Technical details (collapsed by default) -------------------------
        if insights is not None:
            with st.expander("Technical details - ML metrics and raw data patterns", expanded=False):
                st.markdown("**Data overview**")
                st.json(insights.data_summary)

                st.markdown("**Test ML results**")
                st.json(insights.test_ml_results.get("metrics", {}))
                samples = insights.test_ml_results.get("sample_predictions", [])
                if samples:
                    st.dataframe(pd.DataFrame(samples), use_container_width=True)

                st.markdown("**Feature values at peak vs low demand**")
                col_peak_f, col_low_f = st.columns(2)
                with col_peak_f:
                    st.markdown("At peak predictions")
                    st.json(insights.feature_profiles.get("at_peak_predictions", {}))
                with col_low_f:
                    st.markdown("At low predictions")
                    st.json(insights.feature_profiles.get("at_low_predictions", {}))

                col_peak, col_low = st.columns(2)
                with col_peak:
                    st.markdown("**Peak prediction windows**")
                    st.dataframe(pd.DataFrame(insights.peak_periods), use_container_width=True)
                with col_low:
                    st.markdown("**Low prediction windows**")
                    st.dataframe(pd.DataFrame(insights.low_periods), use_container_width=True)

                if insights.temporal_summary:
                    st.markdown("**Temporal patterns**")
                    st.json(insights.temporal_summary)
