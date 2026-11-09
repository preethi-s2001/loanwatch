"""
Manager Dashboard - Executive Portfolio Health, Delinquency Trends & Workload Allocation
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.auth import PAGE_PERMISSIONS, require_login
from src.db import get_collector_workload_metrics
from src.ui_helpers import (
    format_inr,
    load_current_scored_loans,
    load_historical_panel,
    render_dataframe,
    render_disclaimer_banner,
)

st.set_page_config(page_title="Manager Dashboard | LoanWatch", page_icon="👔", layout="wide")

# Enforce role-based access control
current_user = require_login(PAGE_PERMISSIONS["1_Manager_Dashboard"])

st.title("👔 Portfolio Manager Dashboard")
st.markdown("High-level portfolio health, delinquency trends, risk tier concentration, and collection team workload.")

render_disclaimer_banner()

# Load Scored Portfolio and Historical Panel
df_scored = load_current_scored_loans()
df_hist = load_historical_panel()

if df_scored.empty:
    st.warning("No active loan data found in the system. Please upload a loan dataset in the 'Upload Data' page.")
    st.stop()

# 1. Executive Top-Level KPIs
st.markdown("### 📈 Portfolio Executive KPIs")

total_loans = len(df_scored)
total_exposure = df_scored["outstanding_amount"].sum()
high_risk_loans = (df_scored["risk_bucket"] == "High Risk").sum()
dpd_30_plus = (df_scored["days_past_due"] >= 30).sum()
dpd_60_plus = (df_scored["days_past_due"] >= 60).sum()
completed_followups = df_scored["followup_status"].isin(["Completed", "Paid"]).sum()
completion_rate = round((completed_followups / total_loans * 100), 1) if total_loans > 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Active Loans", f"{total_loans:,}")
k2.metric("Total Outstanding (₹)", format_inr(total_exposure))
k3.metric("High-Risk Accounts", f"{high_risk_loans:,}", delta=f"{high_risk_loans/total_loans*100:.1f}% of total", delta_color="inverse")
k4.metric("30+ DPD (SMA-1 & SMA-2)", f"{dpd_30_plus:,}", delta=f"{dpd_30_plus/total_loans*100:.1f}% delinquency", delta_color="inverse")
k5.metric("60+ DPD (Critical)", f"{dpd_60_plus:,}", delta="Immediate field outreach", delta_color="inverse")

st.divider()

# 2. Visualizations: Risk Distribution & DPD Buckets
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("#### 🎯 Active Portfolio Risk Tier Distribution")
    risk_summary = df_scored["risk_bucket"].value_counts().reset_index()
    risk_summary.columns = ["Risk Tier", "Account Count"]

    fig_risk = px.pie(
        risk_summary,
        names="Risk Tier",
        values="Account Count",
        color="Risk Tier",
        color_discrete_map={
            "High Risk": "#d9534f",
            "Medium Risk": "#f39c12",
            "Low Risk": "#27ae60",
        },
        hole=0.45,
    )
    fig_risk.update_traces(textinfo="percent+label", textfont_size=13)
    fig_risk.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_risk, use_container_width=True)

with col_chart2:
    st.markdown("#### ⏳ Delinquency Status Breakdown")
    # Define DPD categories
    df_scored["dpd_stage"] = pd.cut(
        df_scored["days_past_due"],
        bins=[-1, 0, 29, 59, 89],
        labels=["Current (0 DPD)", "SMA-0 (1-29 DPD)", "SMA-1 (30-59 DPD)", "SMA-2 (60-89 DPD)"]
    )
    stage_summary = df_scored["dpd_stage"].value_counts().reindex(
        ["Current (0 DPD)", "SMA-0 (1-29 DPD)", "SMA-1 (30-59 DPD)", "SMA-2 (60-89 DPD)"]
    ).fillna(0).reset_index()
    stage_summary.columns = ["Delinquency Stage", "Count"]

    fig_stage = px.bar(
        stage_summary,
        x="Delinquency Stage",
        y="Count",
        color="Delinquency Stage",
        color_discrete_sequence=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"],
        text="Count",
    )
    fig_stage.update_traces(textposition="outside")
    fig_stage.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=320)
    st.plotly_chart(fig_stage, use_container_width=True)

st.divider()

# 3. Longitudinal Trends: Overdue & Delinquency Migration Over Time
st.markdown("### 📊 Historical Portfolio Delinquency Trends (36-Month Observation Timeline)")

if not df_hist.empty and "observation_date" in df_hist.columns:
    df_hist["obs_dt"] = pd.to_datetime(df_hist["observation_date"])
    monthly_trend = df_hist.groupby("obs_dt").agg(
        total_accounts=("loan_id", "count"),
        dpd_30_count=("days_past_due", lambda x: (x >= 30).sum()),
        dpd_60_count=("days_past_due", lambda x: (x >= 60).sum()),
        avg_days_late=("days_late", "mean"),
    ).reset_index()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_trend["obs_dt"],
        y=monthly_trend["total_accounts"],
        name="Active Accounts",
        mode="lines",
        line=dict(color="#2c3e50", width=2, dash="dot"),
        yaxis="y2"
    ))
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["obs_dt"],
        y=monthly_trend["dpd_30_count"],
        name="30+ DPD Overdue",
        marker_color="#f39c12"
    ))
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["obs_dt"],
        y=monthly_trend["dpd_60_count"],
        name="60+ DPD Overdue",
        marker_color="#e74c3c"
    ))

    fig_trend.update_layout(
        title="Monthly Delinquency Trajectory (30+ and 60+ DPD Accounts)",
        xaxis_title="Observation Cycle",
        yaxis_title="Delinquent Accounts Count",
        yaxis2=dict(
            title="Total Active Portfolio",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Historical monthly timeline not loaded.")

st.divider()

# 4. Collector Workload Allocation & Follow-Up Performance
st.markdown("### 👥 Collector Workload & Follow-Up Progress")

collector_df = get_collector_workload_metrics()

if not collector_df.empty:
    col_kpi1, col_kpi2 = st.columns([1, 1])

    with col_kpi1:
        st.markdown("#### Operational Follow-Up Status by Collector")
        fig_coll = px.bar(
            collector_df,
            x="collector_id",
            y=["pending_count", "in_progress_count", "completed_count"],
            title="Caseload Status Breakdown per Collector",
            labels={"value": "Accounts", "collector_id": "Collection Officer", "variable": "Status"},
            color_discrete_map={
                "pending_count": "#7f8c8d",
                "in_progress_count": "#f39c12",
                "completed_count": "#27ae60",
            },
            barmode="stack",
        )
        fig_coll.update_layout(height=340, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_coll, use_container_width=True)

    with col_kpi2:
        st.markdown("#### High-Risk Exposure Allocation")
        fig_exp = px.bar(
            collector_df,
            x="collector_id",
            y="total_outstanding",
            color="high_risk_count",
            title="Outstanding Portfolio (₹) Managed per Collector",
            labels={"collector_id": "Collector ID", "total_outstanding": "Total Outstanding (₹)", "high_risk_count": "High Risk Accounts"},
            color_continuous_scale="Reds",
        )
        fig_exp.update_layout(height=340, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_exp, use_container_width=True)

    # Detailed Table
    st.markdown("#### Detailed Collection Team Performance Summary")
    disp_coll = collector_df.copy()
    disp_coll["total_outstanding"] = disp_coll["total_outstanding"].apply(format_inr)
    disp_coll.columns = [
        "Collector ID", "Total Assigned", "High Risk", "Pending", "In-Progress", "Completed", "Outstanding Balance"
    ]
    render_dataframe(disp_coll, width="stretch", hide_index=True)
