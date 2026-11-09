"""
Tableau Export Pipeline for LoanWatch NBFC Early Warning System.

Generates four denormalized, ready-to-visualize CSV datasets into the tableau/ directory:
1. portfolio_overview.csv: Point-in-time loan-level active portfolio snapshot.
2. risk_distribution.csv: Aggregated risk tiers, probability deciles, and loss exposure.
3. overdue_analysis.csv: Longitudinal 36-month timeline tracking 30+ DPD and forward defaults.
4. collector_performance.csv: Caseload allocation, follow-up statuses, and completion rates.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    DB_PATH,
    PROCESSED_DATA_PATH,
    SAMPLE_DATA_PATH,
    TABLEAU_DIR,
)
from src.db import get_all_scored_loans_with_status, get_collector_workload_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_portfolio_overview(df_scored: pd.DataFrame, output_dir: Path) -> Path:
    """Exports denormalized loan-level portfolio snapshot."""
    df_out = df_scored.copy()

    # DPD stages
    df_out["dpd_stage"] = pd.cut(
        df_out["days_past_due"],
        bins=[-1, 0, 29, 59, 89],
        labels=["Current (0 DPD)", "SMA-0 (1-29 DPD)", "SMA-1 (30-59 DPD)", "SMA-2 (60-89 DPD)"]
    )

    df_out["foir_pct"] = (df_out["emi_amount"] / df_out["monthly_income"] * 100).round(2)
    df_out["outstanding_ratio_pct"] = (df_out["outstanding_amount"] / df_out["loan_amount"] * 100).round(2)

    cols = [
        "loan_id",
        "observation_date",
        "collector_id",
        "loan_amount",
        "outstanding_amount",
        "emi_amount",
        "monthly_income",
        "days_past_due",
        "dpd_stage",
        "days_late",
        "bounced_emi_count",
        "foir_pct",
        "outstanding_ratio_pct",
        "probability",
        "risk_bucket",
        "followup_status",
        "suggested_action",
    ]
    available_cols = [c for c in cols if c in df_out.columns]
    dest = output_dir / "portfolio_overview.csv"
    df_out[available_cols].to_csv(dest, index=False)
    logger.info("Exported Portfolio Overview to %s (%d rows)", dest, len(df_out))
    return dest


def export_risk_distribution(df_scored: pd.DataFrame, output_dir: Path) -> Path:
    """Exports risk tier and probability decile aggregations."""
    df = df_scored.copy()

    # Create 10 probability deciles
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    df["probability_decile"] = pd.cut(df["probability"], bins=bins, labels=labels, include_lowest=True)

    agg_df = df.groupby(["risk_bucket", "probability_decile"], observed=False).agg(
        account_count=("loan_id", "count"),
        total_outstanding=("outstanding_amount", "sum"),
        avg_probability=("probability", "mean"),
        avg_dpd=("days_past_due", "mean"),
    ).reset_index()

    agg_df["total_outstanding"] = agg_df["total_outstanding"].round(2)
    agg_df["avg_probability"] = agg_df["avg_probability"].round(4)
    agg_df["avg_dpd"] = agg_df["avg_dpd"].round(1)

    dest = output_dir / "risk_distribution.csv"
    agg_df.to_csv(dest, index=False)
    logger.info("Exported Risk Distribution to %s (%d rows)", dest, len(agg_df))
    return dest


def export_overdue_analysis(panel_path: Path, output_dir: Path) -> Path:
    """Exports longitudinal 36-month timeline tracking delinquency migration."""
    df_panel = pd.read_csv(panel_path)
    df_panel["obs_dt"] = pd.to_datetime(df_panel["observation_date"])

    monthly = df_panel.groupby("obs_dt").agg(
        observation_date=("observation_date", "first"),
        total_active_accounts=("loan_id", "count"),
        total_outstanding_portfolio=("outstanding_amount", "sum"),
        current_0_dpd_count=("days_past_due", lambda x: (x == 0).sum()),
        sma0_count=("days_past_due", lambda x: ((x >= 1) & (x < 30)).sum()),
        sma1_count=("days_past_due", lambda x: ((x >= 30) & (x < 60)).sum()),
        sma2_count=("days_past_due", lambda x: ((x >= 60) & (x < 90)).sum()),
        dpd_30_plus_count=("days_past_due", lambda x: (x >= 30).sum()),
        forward_defaults_count=("default_next_3_months", "sum") if "default_next_3_months" in df_panel.columns else ("days_past_due", lambda x: 0),
        avg_days_late=("days_late", "mean"),
    ).reset_index(drop=True)

    monthly["total_outstanding_portfolio"] = monthly["total_outstanding_portfolio"].round(2)
    monthly["dpd_30_plus_rate"] = (monthly["dpd_30_plus_count"] / monthly["total_active_accounts"] * 100).round(2)
    monthly["forward_default_rate"] = (monthly["forward_defaults_count"] / monthly["total_active_accounts"] * 100).round(2)
    monthly["avg_days_late"] = monthly["avg_days_late"].round(2)

    dest = output_dir / "overdue_analysis.csv"
    monthly.to_csv(dest, index=False)
    logger.info("Exported Overdue Analysis to %s (%d monthly cycles)", dest, len(monthly))
    return dest


def export_collector_performance(output_dir: Path, db_path: Optional[Path] = None) -> Path:
    """Exports collector casework workload, follow-up progress, and completion rates."""
    df_scored = get_all_scored_loans_with_status(db_path=db_path)

    collector_agg = df_scored.groupby("collector_id").agg(
        total_assigned=("loan_id", "count"),
        high_risk_assigned=("risk_bucket", lambda x: (x == "High Risk").sum()),
        medium_risk_assigned=("risk_bucket", lambda x: (x == "Medium Risk").sum()),
        low_risk_assigned=("risk_bucket", lambda x: (x == "Low Risk").sum()),
        pending_followups=("followup_status", lambda x: (x == "Pending").sum()),
        promised_to_pay=("followup_status", lambda x: (x == "Contacted – Promised to Pay").sum()),
        no_answer=("followup_status", lambda x: (x == "Contacted – No Answer").sum()),
        completed_followups=("followup_status", lambda x: x.isin(["Completed", "Paid"]).sum()),
        total_outstanding_managed=("outstanding_amount", "sum"),
        high_risk_outstanding=("outstanding_amount", lambda s: s[df_scored.loc[s.index, "risk_bucket"] == "High Risk"].sum()),
    ).reset_index()

    collector_agg["completion_rate_pct"] = (
        collector_agg["completed_followups"] / collector_agg["total_assigned"] * 100
    ).round(2)
    collector_agg["total_outstanding_managed"] = collector_agg["total_outstanding_managed"].round(2)
    collector_agg["high_risk_outstanding"] = collector_agg["high_risk_outstanding"].round(2)

    dest = output_dir / "collector_performance.csv"
    collector_agg.to_csv(dest, index=False)
    logger.info("Exported Collector Performance to %s (%d collectors)", dest, len(collector_agg))
    return dest


def generate_all_tableau_exports():
    """Generates all 4 Tableau export CSVs."""
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    df_scored = get_all_scored_loans_with_status(db_path=DB_PATH)

    export_portfolio_overview(df_scored, TABLEAU_DIR)
    export_risk_distribution(df_scored, TABLEAU_DIR)

    panel_file = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else SAMPLE_DATA_PATH
    export_overdue_analysis(panel_file, TABLEAU_DIR)
    export_collector_performance(TABLEAU_DIR, db_path=DB_PATH)
    logger.info("All Tableau exports generated successfully in %s", TABLEAU_DIR)


if __name__ == "__main__":
    generate_all_tableau_exports()
