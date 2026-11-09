"""
Collector Worklist Page - Prioritised Daily Queue, Reason Flagged & Interactive Follow-up Logging
"""

import streamlit as st
import pandas as pd

from src.auth import (
    CAN_WRITE_FOLLOWUPS_ROLES,
    PAGE_PERMISSIONS,
    can_write_followups,
    require_login,
)
from src.config import COLLECTOR_IDS, FOLLOWUP_STATUSES
from src.db import add_or_update_followup, get_loan_followup_history
from src.ui_helpers import format_inr, render_dataframe, render_disclaimer_banner
from src.worklist import get_collector_kpis, get_prioritized_worklist

st.set_page_config(page_title="Collector Worklist | LoanWatch", page_icon="📋", layout="wide")

# Enforce role-based access control
current_user = require_login(PAGE_PERMISSIONS["4_Collector_Worklist"])

st.title("📋 Daily Collector Worklist & Action Queue")
st.markdown("Prioritised account outreach queue ranked by calibrated default risk and outstanding financial exposure.")

render_disclaimer_banner()

# 1. Collector Selector / Isolation
is_collector = current_user.get("role") == "collector"

if is_collector:
    selected_collector = current_user["collector_id"]
    col_sel2, col_sel3 = st.columns(2)
else:
    col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1.2, 1.2])
    with col_sel1:
        selected_collector = st.selectbox(
            "Select Collection Executive:",
            sorted(COLLECTOR_IDS),
            index=0,
            help="Select executive ID to view assigned daily queue.",
        )

with col_sel2:
    status_filter = st.selectbox(
        "Queue Status Filter:",
        ["All", "Active Pending", "Pending", "Contacted – Promised to Pay", "Contacted – No Answer", "Escalated", "Completed", "Paid"],
        index=1,  # Default to Active Pending
        help="'Active Pending' filters out already completed/paid accounts.",
    )

with col_sel3:
    tier_filter = st.selectbox(
        "Risk Tier Filter:",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        index=0,
    )

# 2. Executive KPI Cards for Selected Collector
kpis = get_collector_kpis(selected_collector)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Assigned Accounts", f"{kpis['total_assigned']:,}")
m2.metric("Total Outstanding (₹)", format_inr(kpis["total_outstanding"]))
m3.metric("High-Risk Accounts", f"{kpis['high_risk_count']:,}", delta="Priority field focus", delta_color="inverse")
m4.metric("Pending Follow-ups", f"{kpis['pending_count']:,}", delta="Pending action", delta_color="inverse")
m5.metric("Completed Accounts", f"{kpis['completed_count']:,}", delta=f"{kpis['completion_pct']}% completed")

st.progress(min(1.0, kpis["completion_pct"] / 100.0), text=f"Daily Outreach Progress: {kpis['completion_pct']}% Completed")

st.divider()

# 3. Fetch Prioritized Queue
worklist_df = get_prioritized_worklist(
    collector_id=selected_collector,
    risk_bucket=tier_filter,
    status_filter=status_filter,
)

if worklist_df.empty:
    st.info(f"No accounts match the current filter criteria for collector **{selected_collector}**.")
    st.stop()

# 4. Interactive Follow-up Action Workspace
st.markdown("### ✍️ Log Borrower Interaction & Outreach Status")

# Account selector
account_options = [
    f"Rank #{row['priority_rank']} | {row['loan_id']} | Risk: {row['probability']*100:.1f}% | Out: {format_inr(row['outstanding_amount'])} | Status: {row['followup_status']}"
    for _, row in worklist_df.iterrows()
]
selected_account_idx = st.selectbox(
    "Select an account from the priority queue to take action:",
    range(len(account_options)),
    format_func=lambda idx: account_options[idx],
)

selected_account = worklist_df.iloc[selected_account_idx]

# Display Loan Context Card
with st.container():
    st.markdown(f"#### Account File: `{selected_account['loan_id']}` (Priority Rank #{selected_account['priority_rank']})")

    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    c_info1.metric("Outstanding Balance", format_inr(selected_account["outstanding_amount"]))
    c_info2.metric("Monthly EMI", format_inr(selected_account["emi_amount"]))
    c_info3.metric("Days Past Due (DPD)", f"{selected_account['days_past_due']} days")
    c_info4.metric("Default Risk Score", f"{selected_account['probability']*100:.1f}% ({selected_account['risk_bucket']})")

    # Warning Signs & Action Guidance
    col_reasons, col_action = st.columns(2)
    with col_reasons:
        st.markdown("**⚠️ Warning Signs Flagged by Model:**")
        signs = selected_account.get("warning_signs", [])
        if isinstance(signs, list):
            for s in signs:
                st.markdown(f"- {s}")
        else:
            st.markdown(f"- {signs}")

    with col_action:
        st.markdown("**💡 Recommended Outreach Protocol:**")
        st.info(f"**Action:** {selected_account['suggested_action']}")
        st.caption("Outreach must remain respectful and in compliance with RBI fair lending principles.")

    # Form to Submit Update (Write-permission restricted)
    if not can_write_followups(current_user.get("role", "")):
        st.info("ℹ️ **Read-Only Mode:** Status updates and follow-up logging are disabled for your role.")
    else:
        with st.form("followup_form"):
            st.markdown("**Update Outreach Status & Log Call/Visit Notes:**")
            f_col1, f_col2 = st.columns([1, 2])

            with f_col1:
                current_status = selected_account["followup_status"]
                default_status_idx = FOLLOWUP_STATUSES.index(current_status) if current_status in FOLLOWUP_STATUSES else 0
                new_status = st.selectbox("Update Status:", FOLLOWUP_STATUSES, index=default_status_idx)

            with f_col2:
                new_notes = st.text_area(
                    "Interaction Notes (e.g., spoke with borrower, promised payment date, settlement offer):",
                    placeholder="Enter detailed summary of conversation or visit outcome...",
                    height=70,
                )

            submit_btn = st.form_submit_button("💾 Save Follow-up Update", type="primary")

            if submit_btn:
                if not new_notes.strip():
                    st.error("Please enter brief interaction notes before saving.")
                else:
                    fid = add_or_update_followup(
                        loan_id=selected_account["loan_id"],
                        collector_id=selected_collector,
                        status=new_status,
                        notes=new_notes.strip(),
                        updated_by=current_user["user_id"],
                    )
                    st.success(f"✅ Follow-up log #{fid} saved for loan `{selected_account['loan_id']}`! Status updated to **{new_status}**.")
                    st.cache_data.clear()
                    st.rerun()

st.divider()

# 5. Full Worklist Table Display
st.markdown("### 📋 Prioritised Daily Worklist Table")
st.caption("Ordered strictly by Default Probability (DESC) with Outstanding Amount (DESC) as tie-breaker.")

# Format display
df_table = worklist_df.copy()
df_table["Formatted Outstanding"] = df_table["outstanding_amount"].apply(format_inr)
df_table["Risk (%)"] = (df_table["probability"] * 100).round(1)

table_cols = [
    "priority_rank",
    "loan_id",
    "Risk (%)",
    "risk_bucket",
    "Formatted Outstanding",
    "days_past_due",
    "followup_status",
    "suggested_action",
    "latest_notes",
]

df_table_view = df_table[table_cols].rename(columns={
    "priority_rank": "Rank",
    "loan_id": "Loan ID",
    "risk_bucket": "Tier",
    "Formatted Outstanding": "Outstanding (₹)",
    "days_past_due": "DPD",
    "followup_status": "Follow-Up Status",
    "suggested_action": "Suggested Action",
    "latest_notes": "Latest Notes",
})

render_dataframe(
    df_table_view,
    width="stretch",
    hide_index=True,
    column_config={
        "Risk (%)": st.column_config.ProgressColumn(
            "Default Risk (%)",
            min_value=0,
            max_value=100,
            format="%.1f%%"
        ),
        "Latest Notes": st.column_config.TextColumn("Latest Notes", width="medium"),
    }
)
