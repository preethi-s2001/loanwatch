"""
UI Utilities and Formatting Helpers for LoanWatch Streamlit Application.
Provides Indian Rupee (INR) digit grouping, responsible-use disclaimers,
badge HTML helpers, and cached data loaders.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
import streamlit as st

from src.config import (
    DB_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RISK_TIERS,
    SAMPLE_DATA_PATH,
)
from src.db import (
    get_all_scored_loans_with_status,
    init_db,
    save_scored_loans,
)


def render_dataframe(df: pd.DataFrame, **kwargs) -> Any:
    """
    Renders st.dataframe supporting both Streamlit < 1.40 and >= 1.40 without TypeErrors.
    In Streamlit >= 1.40, use_container_width is deprecated; width='stretch' is preferred.
    In Streamlit < 1.40, width='stretch' causes TypeError, so use_container_width=True is used.
    """
    st_version = tuple(map(int, st.__version__.split(".")[:2]))
    if st_version >= (1, 40):
        if kwargs.pop("use_container_width", None) is not None or "width" not in kwargs:
            kwargs["width"] = "stretch"
    else:
        if kwargs.get("width") == "stretch":
            kwargs.pop("width")
            kwargs["use_container_width"] = True
        elif "width" not in kwargs and "use_container_width" not in kwargs:
            kwargs["use_container_width"] = True
    return st.dataframe(df, **kwargs)


def format_inr(number: float) -> str:
    """
    Formats a numeric value into the Indian currency system (₹) with comma grouping.
    e.g., 1234567.89 -> ₹12,34,567.89
    """
    if pd.isna(number) or number is None:
        return "₹0.00"

    num = float(number)
    sign = "-" if num < 0 else ""
    num = abs(num)

    s = f"{num:.2f}"
    int_part, dec_part = s.split(".")

    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        last3 = int_part[-3:]
        remaining = int_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        groups.append(last3)
        formatted_int = ",".join(groups)

    return f"{sign}₹{formatted_int}.{dec_part}"


def render_disclaimer_banner():
    """Renders the persistent ethical and responsible-use banner."""
    st.info(
        "🛡️ **Responsible AI & Fair Practices Notice:** Early warning probabilities and risk tiers are advisory "
        "decision-support indicators for human collection officers. **A high risk score is not proof of default.** "
        "All customer communication must strictly comply with the RBI Fair Practices Code: maintain respectful dialogue, "
        "adhere to permitted contact hours (08:00–19:00), and never engage in automated harassment."
    )


def get_risk_badge(bucket: str) -> str:
    """Returns Markdown/HTML badge styling for risk tiers."""
    tier_colors = {
        "High Risk": ("#d9534f", "#ffffff"),
        "Medium Risk": ("#f39c12", "#ffffff"),
        "Low Risk": ("#27ae60", "#ffffff"),
    }
    bg, fg = tier_colors.get(bucket, ("#7f8c8d", "#ffffff"))
    return f"<span style='background-color:{bg}; color:{fg}; padding:3px 10px; border-radius:12px; font-weight:600; font-size:12px;'>{bucket}</span>"


def get_status_badge(status: str) -> str:
    """Returns HTML badge for follow-up workflow statuses."""
    status_colors = {
        "Pending": ("#7f8c8d", "#ffffff"),
        "Contacted – No Answer": ("#e67e22", "#ffffff"),
        "Contacted – Promised to Pay": ("#2980b9", "#ffffff"),
        "Paid": ("#27ae60", "#ffffff"),
        "Completed": ("#16a085", "#ffffff"),
        "Escalated": ("#c0392b", "#ffffff"),
    }
    bg, fg = status_colors.get(status, ("#95a5a6", "#ffffff"))
    return f"<span style='background-color:{bg}; color:{fg}; padding:3px 8px; border-radius:8px; font-size:11px;'>{status}</span>"


@st.cache_resource
def load_trained_model():
    """Loads and caches the serialized model."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def ensure_database_seeded(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Ensures loanwatch.db exists and contains scored active loans.
    If database or loans_scored is empty, automatically scores sample_loans.csv
    and persists records so the application starts with live data on first run.
    """
    init_db(db_path)
    df_existing = get_all_scored_loans_with_status(db_path=db_path)
    if df_existing.empty and SAMPLE_DATA_PATH.exists():
        from src.predict import score_active_loans
        df_sample = pd.read_csv(SAMPLE_DATA_PATH)
        df_scored = score_active_loans(df_sample)
        if not df_scored.empty:
            save_scored_loans(df_scored, db_path=db_path, clear_existing=True)
            df_existing = get_all_scored_loans_with_status(db_path=db_path)
    return df_existing


@st.cache_data(ttl=60)
def load_current_scored_loans(collector_id: Optional[str] = None):
    """Loads scored loans joined with follow-up statuses from SQLite, auto-seeding if empty."""
    ensure_database_seeded(db_path=DB_PATH)
    if collector_id and collector_id != "ALL":
        return get_all_scored_loans_with_status(collector_id=collector_id, db_path=DB_PATH)
    return get_all_scored_loans_with_status(db_path=DB_PATH)


@st.cache_data
def load_historical_panel(collector_id: Optional[str] = None):
    """Loads the historical loan panel for longitudinal analysis, optionally filtered by collector."""
    df = pd.DataFrame()
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    elif SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(SAMPLE_DATA_PATH)
    if not df.empty and collector_id and collector_id != "ALL" and "collector_id" in df.columns:
        df = df[df["collector_id"] == collector_id].reset_index(drop=True)
    return df
