"""
Worklist Prioritization and Queue Allocation Engine for LoanWatch.

Prioritization Logic:
1. Primary Sort: Calibrated default probability (DESC).
2. Tie-Breaker: Outstanding exposure amount in INR (DESC).
Ensures collections executives target accounts with highest expected loss first.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import DB_PATH
from src.db import get_all_scored_loans_with_status


def get_prioritized_worklist(
    collector_id: Optional[str] = None,
    risk_bucket: Optional[str] = None,
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retrieves and filters the operational collection queue.
    Applies strict priority ordering: probability DESC, outstanding_amount DESC.
    """
    df = get_all_scored_loans_with_status(db_path=db_path, collector_id=collector_id)

    if df.empty:
        return df

    # 1. Filter by Risk Tier
    if risk_bucket and risk_bucket != "All":
        df = df[df["risk_bucket"] == risk_bucket]

    # 2. Filter by Follow-up Status
    if status_filter and status_filter != "All":
        if status_filter == "Active Pending":
            # Shows accounts requiring immediate action (Pending or In-Progress)
            df = df[~df["followup_status"].isin(["Completed", "Paid"])]
        else:
            df = df[df["followup_status"] == status_filter]

    # 3. Filter by Search Query (loan_id substring match)
    if search_query and search_query.strip():
        q = search_query.strip().upper()
        df = df[df["loan_id"].str.upper().str.contains(q)]

    # 4. Strict Dual-Key Sort: Probability DESC, Outstanding Amount DESC
    df = df.sort_values(
        by=["probability", "outstanding_amount"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Assign 1-based Priority Rank
    df["priority_rank"] = range(1, len(df) + 1)

    return df


def get_collector_kpis(
    collector_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes key performance indicators for a single collector's active workload.
    """
    df = get_all_scored_loans_with_status(db_path=db_path, collector_id=collector_id)

    if df.empty:
        return {
            "total_assigned": 0,
            "high_risk_count": 0,
            "pending_count": 0,
            "completed_count": 0,
            "completion_pct": 0.0,
            "total_outstanding": 0.0,
        }

    total = len(df)
    completed = int(df["followup_status"].isin(["Completed", "Paid"]).sum())
    pending = int((df["followup_status"] == "Pending").sum())
    high_risk = int((df["risk_bucket"] == "High Risk").sum())
    total_out = float(df["outstanding_amount"].sum())
    completion_rate = round((completed / total * 100), 1) if total > 0 else 0.0

    return {
        "total_assigned": total,
        "high_risk_count": high_risk,
        "pending_count": pending,
        "completed_count": completed,
        "completion_pct": completion_rate,
        "total_outstanding": total_out,
    }
