"""
Upload Data Page - File Upload, Validation Report & One-Click Prediction Ingestion
"""

import io
from pathlib import Path
import pandas as pd
import streamlit as st

from src.auth import PAGE_PERMISSIONS, require_login
from src.config import (
    DIRTY_SAMPLE_DATA_PATH,
    SAMPLE_DATA_PATH,
)
from src.db import save_scored_loans
from src.predict import score_active_loans
from src.ui_helpers import format_inr, render_dataframe, render_disclaimer_banner
from src.validation import validate_loan_data

st.set_page_config(page_title="Upload Data | LoanWatch", page_icon="📤", layout="wide")

# Enforce role-based access control
current_user = require_login(PAGE_PERMISSIONS["2_Upload_Data"])

st.title("📤 Portfolio Ingestion & Data Validation Engine")
st.markdown("Upload monthly loan-month panel or active portfolio CSV files for automated schema checks, anomaly scrubbing, and early-warning risk scoring.")

render_disclaimer_banner()

# 1. Download Sample Templates
st.markdown("### 📥 Download Verification Templates")
st.markdown("Use these benchmark files to test validator behavior on clean vs anomaly-injected datasets:")

c_down1, c_down2 = st.columns(2)

with c_down1:
    if SAMPLE_DATA_PATH.exists():
        with open(SAMPLE_DATA_PATH, "rb") as f:
            clean_bytes = f.read()
        st.download_button(
            label="📄 Download Clean Benchmark Dataset (CSV)",
            data=clean_bytes,
            file_name="sample_loans.csv",
            mime="text/csv",
            help="Clean synthetic panel containing 59,000+ validated records.",
        )

with c_down2:
    if DIRTY_SAMPLE_DATA_PATH.exists():
        with open(DIRTY_SAMPLE_DATA_PATH, "rb") as f:
            dirty_bytes = f.read()
        st.download_button(
            label="⚠️ Download Anomaly-Injected Test File (CSV)",
            data=dirty_bytes,
            file_name="dirty_sample_loans.csv",
            mime="text/csv",
            help="Contains negative loan amounts, missing income, invalid dates, and duplicates to test validator robustness.",
        )

st.divider()

# 2. File Upload Component
st.markdown("### 📂 Upload Active Loan File")
uploaded_file = st.file_uploader(
    "Choose a CSV file containing active loan accounts",
    type=["csv"],
    help="Upload a CSV matching the LoanWatch data dictionary schema.",
)

if uploaded_file is not None:
    try:
        # Load raw file
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"Successfully read `{uploaded_file.name}` ({len(df_uploaded):,} rows, {len(df_uploaded.columns)} columns).")

        # Execute Validation Engine
        with st.spinner("Running schema validation, anomaly detection, and deduplication checks..."):
            report = validate_loan_data(df_uploaded)

        # 3. Validation Summary Card
        st.markdown("### 📋 Validation Audit Report")
        v1, v2, v3, v4 = st.columns(4)

        status_label = "✅ PASSED" if report.is_valid else "❌ ACTION REQUIRED"
        v1.metric("Validation Status", status_label)
        v2.metric("Input Records", f"{report.total_rows_input:,}")
        v3.metric("Clean Records Retained", f"{report.rows_retained:,}")
        v4.metric("Anomalies / Dropped", f"{report.rows_dropped:,}")

        # Display Errors & Warnings
        if report.errors:
            with st.expander(f"🚨 Validation Errors Detected ({len(report.errors)})", expanded=True):
                for err in report.errors:
                    st.error(f"• {err}")

        if report.warnings:
            with st.expander(f"⚠️ Operational Warnings & Notes ({len(report.warnings)})", expanded=False):
                for w in report.warnings:
                    st.warning(f"• {w}")

        # Preview Cleaned Data
        if not report.cleaned_df.empty:
            st.markdown("#### 👁️ Preview Validated Dataset (First 5 Rows)")
            preview_cols = [
                "loan_id", "observation_date", "loan_amount", "outstanding_amount",
                "emi_amount", "monthly_income", "days_past_due", "days_late",
                "bounced_emi_count", "collector_id"
            ]
            cols_to_show = [c for c in preview_cols if c in report.cleaned_df.columns]
            render_dataframe(report.cleaned_df[cols_to_show].head(5), width="stretch", hide_index=True)

            st.divider()

            # 4. Interactive Prediction Execution
            st.markdown("### ⚡ Early Warning Model Scoring")
            st.markdown("Execute the trained Calibrated Random Forest model on the validated active loan snapshot.")

            if st.button("🚀 Run Risk Prediction on Validated Accounts", type="primary", use_container_width=True):
                with st.spinner("Calculating point-in-time features, calibrating probabilities, and generating plain-English warning signs..."):
                    try:
                        df_scored = score_active_loans(report.cleaned_df)
                        st.session_state["latest_scored_upload"] = df_scored

                        st.success(f"🎉 Successfully scored {len(df_scored):,} active loans!")

                        # Summary of scored batch
                        sc1, sc2, sc3 = st.columns(3)
                        high_count = (df_scored["risk_bucket"] == "High Risk").sum()
                        med_count = (df_scored["risk_bucket"] == "Medium Risk").sum()
                        low_count = (df_scored["risk_bucket"] == "Low Risk").sum()

                        sc1.metric("High Risk Accounts (Field Visit)", f"{high_count:,}", delta=f"{high_count/len(df_scored)*100:.1f}%")
                        sc2.metric("Medium Risk Accounts (Call)", f"{med_count:,}", delta=f"{med_count/len(df_scored)*100:.1f}%")
                        sc3.metric("Low Risk Accounts (Reminder)", f"{low_count:,}", delta=f"{low_count/len(df_scored)*100:.1f}%")

                    except Exception as pred_err:
                        st.error(f"Error during model scoring: {str(pred_err)}")

            # 5. Database Commit Option
            if "latest_scored_upload" in st.session_state and not st.session_state["latest_scored_upload"].empty:
                st.markdown("#### 💾 Commit to Live Portfolio Database")
                st.markdown("Save scored records to the persistent SQLite database to update the Collector Worklist and Manager Dashboard.")

                if st.button("📥 Save & Commit Scored Accounts to Database", use_container_width=True):
                    with st.spinner("Persisting records and seeding daily collection queue..."):
                        saved = save_scored_loans(st.session_state["latest_scored_upload"])
                        st.success(f"✅ Successfully committed {saved:,} scored loans to `loanwatch.db`! Daily worklist updated.")
                        st.cache_data.clear()

    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
else:
    st.info("👆 Please upload a loan panel CSV file or download one of the benchmark templates above.")
