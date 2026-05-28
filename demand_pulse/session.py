"""Centralized Streamlit session-state keys and initialization."""

from __future__ import annotations

from enum import Enum
from typing import Any

import streamlit as st


class SessionKeys(str, Enum):
    """Canonical keys for ``st.session_state``."""

    DF_TRAIN = "df_train"
    DF_TEST = "df_test"
    TARGET_COL = "target_col"
    INDEX_COLS = "index_cols"
    FEATURE_COLS = "feature_cols"
    FEATURE_NAMES = "feature_names"
    X_TRAIN = "X_train"
    X_TEST = "X_test"
    Y_TRAIN = "y_train"
    Y_TEST = "y_test"
    DATA_READY = "data_ready"
    HAS_TEST_TARGET = "has_test_target"
    MODEL = "model"
    MODEL_PARAMS = "model_params"
    TRAINING_RESULT = "training_result"
    INSIGHTS = "insights"
    LLM_REPORT = "llm_report"
    TRAINED = "trained"
    LLM_ANALYZED = "llm_analyzed"
    PIPELINE_STAGE = "pipeline_stage"
    DROPPED_ID_COLS = "dropped_id_cols"
    COLUMN_PROFILE = "column_profile"
    TARGET_PROFILE = "target_profile"
    TEST_PREDICTIONS = "test_predictions"


STAGE_IDLE = "idle"
STAGE_PREP = "prep"
STAGE_TRAIN = "train"
STAGE_INSIGHTS = "insights"
STAGE_LLM = "llm"
STAGE_DONE = "done"


def init_session_state() -> None:
    """Initialize defaults once per browser session."""
    defaults: dict[str, Any] = {
        SessionKeys.DF_TRAIN: None,
        SessionKeys.DF_TEST: None,
        SessionKeys.TARGET_COL: None,
        SessionKeys.INDEX_COLS: [],
        SessionKeys.FEATURE_COLS: [],
        SessionKeys.FEATURE_NAMES: [],
        SessionKeys.X_TRAIN: None,
        SessionKeys.X_TEST: None,
        SessionKeys.Y_TRAIN: None,
        SessionKeys.Y_TEST: None,
        SessionKeys.DATA_READY: False,
        SessionKeys.HAS_TEST_TARGET: False,
        SessionKeys.MODEL: None,
        SessionKeys.MODEL_PARAMS: {},
        SessionKeys.TRAINING_RESULT: None,
        SessionKeys.INSIGHTS: None,
        SessionKeys.LLM_REPORT: "",
        SessionKeys.TRAINED: False,
        SessionKeys.LLM_ANALYZED: False,
        SessionKeys.PIPELINE_STAGE: STAGE_IDLE,
        SessionKeys.DROPPED_ID_COLS: [],
        SessionKeys.COLUMN_PROFILE: None,
        SessionKeys.TARGET_PROFILE: None,
        SessionKeys.TEST_PREDICTIONS: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline_artifacts() -> None:
    """Clear model-dependent state when data or configuration changes."""
    st.session_state[SessionKeys.X_TRAIN] = None
    st.session_state[SessionKeys.X_TEST] = None
    st.session_state[SessionKeys.Y_TRAIN] = None
    st.session_state[SessionKeys.Y_TEST] = None
    st.session_state[SessionKeys.DATA_READY] = False
    st.session_state[SessionKeys.HAS_TEST_TARGET] = False
    st.session_state[SessionKeys.MODEL] = None
    st.session_state[SessionKeys.TRAINING_RESULT] = None
    st.session_state[SessionKeys.INSIGHTS] = None
    st.session_state[SessionKeys.LLM_REPORT] = ""
    st.session_state[SessionKeys.TRAINED] = False
    st.session_state[SessionKeys.LLM_ANALYZED] = False
    st.session_state[SessionKeys.PIPELINE_STAGE] = STAGE_IDLE
    st.session_state[SessionKeys.DROPPED_ID_COLS] = []
    st.session_state[SessionKeys.TARGET_PROFILE] = None
    st.session_state[SessionKeys.TEST_PREDICTIONS] = None
