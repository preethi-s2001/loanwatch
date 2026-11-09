"""
LoanWatch – NBFC Early Warning System
Streamlit Main Application Entry Point
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.auth import require_login
from src.config import DB_PATH, SAMPLE_DATA_PATH
from src.ui_helpers import format_inr, load_current_scored_loans, render_disclaimer_banner

st.set_page_config(
    page_title="LoanWatch – NBFC Early Warning System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enforce authentication gatekeeper - unauthenticated users only see the clean login form
current_user = require_login()

# Business Head role is restricted to Manager Dashboard only
if current_user.get("role") == "business_head":
    st.switch_page("pages/1_Manager_Dashboard.py")

# Custom Styling for Authenticated View
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 16px;
        color: #555555;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1e3d59;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .kpi-title {
        font-size: 13px;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown("<div class='main-title'>🏦 LoanWatch – NBFC Early Warning System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Proactive delinquency detection, multi-signal risk scoring, and prioritised collection workflows for retail & MSME credit portfolios.</div>",
    unsafe_allow_html=True,
)

# Persistent Responsible-Use Banner
render_disclaimer_banner()

# Load Scored Portfolio (server-side scoped to collector_id if user is a collector)
is_collector = current_user.get("role") == "collector"
collector_filter = current_user.get("collector_id") if is_collector else None
df_scored = load_current_scored_loans(collector_id=collector_filter)

# Global Portfolio KPIs
portfolio_heading = "### 📊 My Assigned Accounts Snapshot" if is_collector else "### 📊 Active Portfolio Snapshot"
st.markdown(portfolio_heading)

col1, col2, col3, col4, col5 = st.columns(5)

if not df_scored.empty:
    total_active = len(df_scored)
    total_out = df_scored["outstanding_amount"].sum()
    high_risk_count = (df_scored["risk_bucket"] == "High Risk").sum()
    sma2_count = (df_scored["days_past_due"] >= 60).sum()
    completed_count = df_scored["followup_status"].isin(["Completed", "Paid"]).sum()
    completion_rate = round((completed_count / total_active * 100), 1) if total_active > 0 else 0.0

    with col1:
        st.metric("Total Active Loans", f"{total_active:,}")
    with col2:
        st.metric("Outstanding Exposure", format_inr(total_out))
    with col3:
        st.metric("High-Risk Accounts", f"{high_risk_count:,}", delta=f"{high_risk_count/total_active*100:.1f}% of scope", delta_color="inverse")
    with col4:
        st.metric("SMA-2 Accounts (60+ DPD)", f"{sma2_count:,}", delta="Critical recovery zone", delta_color="inverse")
    with col5:
        st.metric("Worklist Completed", f"{completed_count:,}", delta=f"{completion_rate}% closed")
else:
    st.warning("No scored portfolio data found in SQLite database. Upload loan data or initialize with sample dataset.")

st.divider()

# Persona Navigation Guide
st.markdown("### 🧭 Platform Navigation Guide")

if not is_collector:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("""
        #### 👔 For Management & Credit Risk Teams:
        - **[1_Manager_Dashboard](1_Manager_Dashboard):** High-level executive view of portfolio delinquency trends, risk tier distribution, 30/90-day delinquency migration, and collector workload allocation.
        - **[2_Upload_Data](2_Upload_Data):** Ingest monthly loan snapshots with schema verification, anomaly checks, and instant one-click batch prediction.
        """)
    with col_right:
        st.markdown("""
        #### 📋 For Operations & Collection Executives:
        - **[3_Risk_Prediction](3_Risk_Prediction):** Explore model-predicted default probabilities, filter accounts by risk tiers, review top warning signs, and export scored lists.
        - **[4_Collector_Worklist](4_Collector_Worklist):** Prioritised daily collection queue ranked by calibrated default probability and tie-broken by outstanding exposure; log customer promises and mark cases complete.
        - **[5_Loan_Details](5_Loan_Details):** Single-account deep-dive with historical DPD trajectory charts, borrower underwriting metrics, and chronological follow-up audit trail.
        """)
else:
    st.markdown("""
    #### 📋 My Assigned Collection Tools:
    - **[3_Risk_Prediction](3_Risk_Prediction):** Review predicted default risk and top warning signs for your assigned accounts.
    - **[4_Collector_Worklist](4_Collector_Worklist):** Your daily prioritized outreach queue. Log customer contact updates, promise-to-pay dates, and notes.
    - **[5_Loan_Details](5_Loan_Details):** In-depth single-account borrower history and payment trajectory.
    """)

st.divider()

# System Specifications & Technology Highlights
with st.expander("ℹ️ Architecture & System Highlights", expanded=False):
    st.markdown("""
    - **Role-Based Access Control:** Secure session authentication with strict privilege separation and collector data isolation.
    - **Leakage-Free Feature Engineering:** Point-in-time temporal features computed strictly using data available on or before each monthly observation date ($t \le T$).
    - **Calibrated Posterior Probabilities:** Probabilities reflect empirical failure rates, optimizing recovery workflows.
    - **Dual-Key Worklist Prioritization:** Accounts are ranked by default probability and strictly tie-broken by unamortized balance.
    - **Full SQLite Persistence:** Every collector interaction, promise-to-pay date, and status transition is recorded with user attribution.
    """)
