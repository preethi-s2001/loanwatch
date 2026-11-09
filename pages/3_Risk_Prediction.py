"""
Risk Prediction Page - Active Loans Probability Scoring, Plain-English Warning Signs & Filters
"""

import json
import pandas as pd
import streamlit as st

from src.auth import PAGE_PERMISSIONS, require_login
from src.config import COLLECTOR_IDS
from src.ui_helpers import (
    format_inr,
    load_current_scored_loans,
    render_dataframe,
    render_disclaimer_banner,
)

st.set_page_config(page_title="Risk Prediction | LoanWatch", page_icon="🎯", layout="wide")

# Enforce role-based access control
current_user = require_login(PAGE_PERMISSIONS["3_Risk_Prediction"])

st.title("🎯 Early Warning Risk Predictions")
st.markdown("Calibrated default probabilities ($\hat{p}$), operational risk tiers, plain-English warning signs, and respectful action recommendations for active loans.")

render_disclaimer_banner()

# Load current active scored loans from database (scoped to collector_id for collector role)
is_collector = current_user.get("role") == "collector"
collector_filter = current_user.get("collector_id") if is_collector else None
df_scored = load_current_scored_loans(collector_id=collector_filter)

if df_scored.empty:
    st.warning("No active loan records found in the database. Please visit the 'Upload Data' page to score accounts.")
    st.stop()

# 1. Interactive Filters
st.markdown("### 🔍 Filter Portfolio Accounts")

if is_collector:
    col_f1, col_f3, col_f4 = st.columns([1.2, 1.4, 1.4])
    selected_collector = current_user["collector_id"]
else:
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1.2, 1.2])
    with col_f2:
        collector_options = ["All"] + sorted(COLLECTOR_IDS)
        selected_collector = st.selectbox("Assigned Collector:", collector_options, index=0)

with col_f1:
    bucket_options = ["All", "High Risk", "Medium Risk", "Low Risk"]
    selected_bucket = st.selectbox("Risk Tier:", bucket_options, index=0)

with col_f3:
    prob_range = st.slider(
        "Default Probability Range (%):",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=5.0,
    )

with col_f4:
    search_query = st.text_input("Search Loan ID:", placeholder="e.g. LN_100001").strip().upper()

# Apply Filters
df_filtered = df_scored.copy()

if selected_bucket != "All":
    df_filtered = df_filtered[df_filtered["risk_bucket"] == selected_bucket]

if not is_collector and selected_collector != "All":
    df_filtered = df_filtered[df_filtered["collector_id"] == selected_collector]

min_p, max_p = prob_range[0] / 100.0, prob_range[1] / 100.0
df_filtered = df_filtered[(df_filtered["probability"] >= min_p) & (df_filtered["probability"] <= max_p)]

if search_query:
    df_filtered = df_filtered[df_filtered["loan_id"].str.contains(search_query, na=False)]

# Sort by probability DESC, tie-break by outstanding amount DESC
df_filtered = df_filtered.sort_values(
    by=["probability", "outstanding_amount"],
    ascending=[False, False]
).reset_index(drop=True)

# 2. Results Header & Summary Stats
st.markdown(f"### 📋 Scored Accounts ({len(df_filtered):,} accounts found)")

s1, s2, s3, s4 = st.columns(4)
total_sub_out = df_filtered["outstanding_amount"].sum()
avg_prob = df_filtered["probability"].mean() * 100 if len(df_filtered) > 0 else 0.0
high_sub = (df_filtered["risk_bucket"] == "High Risk").sum()

s1.metric("Accounts Selected", f"{len(df_filtered):,}")
s2.metric("Filtered Outstanding Exposure", format_inr(total_sub_out))
s3.metric("Average Default Probability", f"{avg_prob:.2f}%")
s4.metric("High-Risk Cases", f"{high_sub:,}")

st.divider()

# 3. Interactive Data Table
# Format display columns
df_display = df_filtered.copy()
df_display["Formatted Outstanding"] = df_display["outstanding_amount"].apply(format_inr)
df_display["Formatted EMI"] = df_display["emi_amount"].apply(format_inr)
df_display["Probability (%)"] = (df_display["probability"] * 100).round(2)

# Join warning signs into bullet text for table
df_display["Primary Warning Signs"] = df_display["warning_signs"].apply(
    lambda signs: " | ".join(signs) if isinstance(signs, list) else str(signs)
)

cols_table = [
    "loan_id",
    "risk_bucket",
    "Probability (%)",
    "Formatted Outstanding",
    "days_past_due",
    "collector_id",
    "suggested_action",
    "Primary Warning Signs",
]

# Rename for clean presentation
df_view = df_display[cols_table].rename(columns={
    "loan_id": "Loan ID",
    "risk_bucket": "Risk Tier",
    "days_past_due": "DPD",
    "collector_id": "Collector",
    "suggested_action": "Recommended Outreach Action",
})

# Display with progress bar styling for probability
render_dataframe(
    df_view,
    width="stretch",
    hide_index=True,
    column_config={
        "Probability (%)": st.column_config.ProgressColumn(
            "Default Risk (%)",
            help="Calibrated probability of default within the next 3 months",
            format="%.2f%%",
            min_value=0,
            max_value=100,
        ),
        "Risk Tier": st.column_config.TextColumn(
            "Risk Tier",
            help="High (>=60%), Medium (30-60%), Low (<30%)",
        ),
        "Formatted Outstanding": st.column_config.TextColumn("Outstanding (₹)"),
        "Primary Warning Signs": st.column_config.TextColumn("Model Warning Signs", width="large"),
        "Recommended Outreach Action": st.column_config.TextColumn("Action Guidance", width="medium"),
    },
)

# 4. CSV Download Option
csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Scored Loans (CSV)",
    data=csv_bytes,
    file_name="loanwatch_risk_predictions.csv",
    mime="text/csv",
)
