"""
Loan Details Page - Single Account Deep-Dive, Trajectory Charts & Interaction Audit Trail
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.auth import (
    CAN_WRITE_FOLLOWUPS_ROLES,
    PAGE_PERMISSIONS,
    can_write_followups,
    require_login,
)
from src.db import get_all_scored_loans_with_status, get_loan_followup_history
from src.ui_helpers import (
    format_inr,
    load_historical_panel,
    render_dataframe,
    render_disclaimer_banner,
)

st.set_page_config(page_title="Loan Details | LoanWatch", page_icon="🔍", layout="wide")

# Enforce role-based access control
current_user = require_login(PAGE_PERMISSIONS["5_Loan_Details"])

st.title("🔍 Loan Account Deep-Dive & Delinquency Trajectory")
st.markdown("Detailed borrower profile, longitudinal repayment behavior over time, and complete collection interaction history.")

render_disclaimer_banner()

# Load scored accounts (strictly scoped to collector_id for collector role)
is_collector = current_user.get("role") == "collector"
collector_filter = current_user.get("collector_id") if is_collector else None
df_scored = get_all_scored_loans_with_status(collector_id=collector_filter)
df_hist = load_historical_panel(collector_id=collector_filter)

if df_scored.empty:
    st.warning("No scored accounts available. Please upload loan records on the 'Upload Data' page.")
    st.stop()

# 1. Account Search / Selection
col_s1, col_s2 = st.columns([1.5, 2])

with col_s1:
    search_input = st.text_input("Enter Loan ID to inspect:", placeholder="e.g. LN_100001").strip().upper()

loan_id_list = df_scored["loan_id"].tolist()

if search_input:
    # Strictly enforce that loans belonging to other collectors cannot be accessed
    if search_input not in loan_id_list:
        st.warning("Loan not found")
        st.stop()
    default_loan = search_input
else:
    default_loan = loan_id_list[0] if loan_id_list else ""

with col_s2:
    selected_loan_id = st.selectbox(
        "Or pick from your assigned portfolio:" if is_collector else "Or pick from the active portfolio:",
        loan_id_list,
        index=loan_id_list.index(default_loan) if default_loan in loan_id_list else 0,
    )

loan_record = df_scored[df_scored["loan_id"] == selected_loan_id].iloc[0]

st.divider()

# 2. Borrower Profile & Current Early-Warning State
st.markdown(f"### 👤 Borrower Contract Profile: `{selected_loan_id}`")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Sanctioned Loan Amount", format_inr(loan_record["loan_amount"]))
p2.metric("Current Outstanding Balance", format_inr(loan_record["outstanding_amount"]))
p3.metric("Scheduled Monthly EMI", format_inr(loan_record["emi_amount"]))
p4.metric("Borrower Monthly Income", format_inr(loan_record["monthly_income"]))

foir = (loan_record["emi_amount"] / loan_record["monthly_income"]) * 100
balance_ratio = (loan_record["outstanding_amount"] / loan_record["loan_amount"]) * 100

st.markdown(
    f"**Underwriting Indicators:** Debt-to-Income (FOIR): **{foir:.1f}%** | "
    f"Unamortized Balance Ratio: **{balance_ratio:.1f}%** | "
    f"Assigned Collector: **{loan_record['collector_id']}**"
)

# 3. Model Early-Warning Assessment Box
st.markdown("### 🎯 Early Warning Risk Assessment")

col_gauge, col_signs = st.columns([1.2, 2])

with col_gauge:
    prob_val = float(loan_record["probability"]) * 100

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_val,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": f"<b>{loan_record['risk_bucket']}</b>", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1e3d59"},
            "steps": [
                {"range": [0, 30], "color": "#d4efdf"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_signs:
    st.markdown("#### ⚠️ Plain-English Warning Signs")
    signs = loan_record.get("warning_signs", [])
    if isinstance(signs, list):
        for s in signs:
            st.markdown(f"- **{s}**")
    else:
        st.markdown(f"- **{signs}**")

    st.markdown("#### 💡 Suggested Action")
    st.info(f"**Recommended Channel:** {loan_record['suggested_action']}")
    st.caption("Action recommendations are advisory guidance for human collectors. Harassment is strictly prohibited.")

st.divider()

# 4. Longitudinal Repayment History Chart
st.markdown("### 📈 Historical Repayment & Delinquency Trajectory")

if not df_hist.empty and "loan_id" in df_hist.columns:
    loan_history = df_hist[df_hist["loan_id"] == selected_loan_id].copy()
    if not loan_history.empty:
        loan_history["obs_dt"] = pd.to_datetime(loan_history["observation_date"])
        loan_history = loan_history.sort_values("obs_dt")

        fig_traj = go.Figure()

        # DPD line
        fig_traj.add_trace(go.Scatter(
            x=loan_history["obs_dt"],
            y=loan_history["days_past_due"],
            name="Days Past Due (DPD)",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=7),
        ))

        # Days late bars
        fig_traj.add_trace(go.Bar(
            x=loan_history["obs_dt"],
            y=loan_history["days_late"],
            name="Payment Delay (Days Late)",
            marker_color="#3498db",
            opacity=0.5,
        ))

        fig_traj.update_layout(
            title=f"Historical Delinquency Trajectory for {selected_loan_id}",
            xaxis_title="Monthly Observation Cycle",
            yaxis_title="Days",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("No multi-month historical records found for this specific loan in the panel.")
else:
    st.info("Historical panel dataset not loaded.")

st.divider()

# 5. Chronological Follow-Up Interaction Audit Trail
st.markdown("### 🗂️ Chronological Collector Interaction History")

history_df = get_loan_followup_history(selected_loan_id)

if not history_df.empty:
    disp_hist = history_df[[
        "created_at", "status", "updated_by", "notes", "completed_at"
    ]].rename(columns={
        "created_at": "Timestamp",
        "status": "Interaction Status",
        "updated_by": "Logged By",
        "notes": "Conversation / Field Notes",
        "completed_at": "Completed At",
    })
    render_dataframe(disp_hist, width="stretch", hide_index=True)
else:
    st.info("No prior interaction logs recorded for this account. Go to the 'Collector Worklist' to record follow-up notes.")
