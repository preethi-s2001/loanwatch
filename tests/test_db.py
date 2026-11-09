"""
Unit tests for SQLite database persistence (src/db.py) and worklist logic (src/worklist.py).
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.db import (
    add_or_update_followup,
    get_all_scored_loans_with_status,
    get_collector_workload_metrics,
    get_loan_followup_history,
    init_db,
    save_scored_loans,
)
from src.worklist import get_collector_kpis, get_prioritized_worklist


@pytest.fixture
def temp_db():
    """Provides an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    init_db(temp_path)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


def test_save_and_retrieve_scored_loans(temp_db):
    """Verify storing scored loans and auto-seeding initial follow-ups."""
    sample_df = pd.DataFrame({
        "loan_id": ["LN_TEST_1", "LN_TEST_2"],
        "probability": [0.75, 0.45],
        "risk_bucket": ["High Risk", "Medium Risk"],
        "warning_signs": [
            ["Account is 60 DPD", "2 bounces in last 3m"],
            ["1 delayed payment"],
        ],
        "collector_id": ["COL_101", "COL_102"],
        "loan_amount": [500000.0, 300000.0],
        "outstanding_amount": [420000.0, 180000.0],
        "emi_amount": [15000.0, 9000.0],
        "monthly_income": [35000.0, 40000.0],
        "days_past_due": [60, 20],
        "suggested_action": [
            "Consider in-person follow-up",
            "Initiate telephone check-in",
        ],
    })

    saved_count = save_scored_loans(sample_df, db_path=temp_db)
    assert saved_count == 2

    # Query back
    df_retrieved = get_all_scored_loans_with_status(db_path=temp_db)
    assert len(df_retrieved) == 2
    assert df_retrieved.iloc[0]["loan_id"] == "LN_TEST_1"
    assert df_retrieved.iloc[0]["followup_status"] == "Pending"
    assert isinstance(df_retrieved.iloc[0]["warning_signs"], list)


def test_worklist_prioritization_and_tie_breaking(temp_db):
    """Verify primary sort by probability DESC, and tie-breaking by outstanding_amount DESC."""
    tied_df = pd.DataFrame({
        "loan_id": ["LN_LOW_AMT", "LN_HIGH_AMT"],
        "probability": [0.65, 0.65],  # Same probability!
        "risk_bucket": ["High Risk", "High Risk"],
        "warning_signs": [["Warning A"], ["Warning B"]],
        "collector_id": ["COL_101", "COL_101"],
        "loan_amount": [500000.0, 500000.0],
        "outstanding_amount": [80000.0, 350000.0],  # Second has higher outstanding!
        "emi_amount": [15000.0, 15000.0],
        "monthly_income": [40000.0, 40000.0],
        "days_past_due": [45, 45],
        "suggested_action": ["Action A", "Action B"],
    })

    save_scored_loans(tied_df, db_path=temp_db)

    worklist = get_prioritized_worklist(collector_id="COL_101", db_path=temp_db)
    assert len(worklist) == 2

    # High amount account must be prioritized as Rank 1
    assert worklist.iloc[0]["loan_id"] == "LN_HIGH_AMT"
    assert worklist.iloc[0]["priority_rank"] == 1
    assert worklist.iloc[1]["loan_id"] == "LN_LOW_AMT"
    assert worklist.iloc[1]["priority_rank"] == 2


def test_followup_status_transition_and_history(temp_db):
    """Verify follow-up logging, status updates, completion timestamps, and audit trail."""
    df = pd.DataFrame({
        "loan_id": ["LN_FOLLOWUP_1"],
        "probability": [0.82],
        "risk_bucket": ["High Risk"],
        "warning_signs": [["Multiple bounces"]],
        "collector_id": ["COL_103"],
        "loan_amount": [200000.0],
        "outstanding_amount": [150000.0],
        "emi_amount": [8000.0],
        "monthly_income": [25000.0],
        "days_past_due": [55],
        "suggested_action": ["Consider in-person follow-up"],
    })
    save_scored_loans(df, db_path=temp_db)

    # Initial state should be Pending
    wl = get_all_scored_loans_with_status(db_path=temp_db, collector_id="COL_103")
    assert wl.iloc[0]["followup_status"] == "Pending"

    # Step 1: Collector calls, borrower promises to pay
    add_or_update_followup(
        loan_id="LN_FOLLOWUP_1",
        collector_id="COL_103",
        status="Contacted – Promised to Pay",
        notes="Borrower promised to clear overdue installment by Friday.",
        updated_by="Collector 103",
        db_path=temp_db,
    )

    wl_updated = get_all_scored_loans_with_status(db_path=temp_db, collector_id="COL_103")
    assert wl_updated.iloc[0]["followup_status"] == "Contacted – Promised to Pay"
    assert "Friday" in wl_updated.iloc[0]["latest_notes"]
    assert wl_updated.iloc[0]["completed_at"] is None

    # Step 2: Payment cleared -> Mark Completed
    add_or_update_followup(
        loan_id="LN_FOLLOWUP_1",
        collector_id="COL_103",
        status="Completed",
        notes="Payment received in full. Case closed.",
        updated_by="Collector 103",
        db_path=temp_db,
    )

    wl_completed = get_all_scored_loans_with_status(db_path=temp_db, collector_id="COL_103")
    assert wl_completed.iloc[0]["followup_status"] == "Completed"
    assert wl_completed.iloc[0]["completed_at"] is not None

    # Verify complete history audit trail
    history = get_loan_followup_history(loan_id="LN_FOLLOWUP_1", db_path=temp_db)
    assert len(history) == 3  # Initial Pending + Promised to Pay + Completed
    assert history.iloc[0]["status"] == "Completed"


def test_collector_kpis_and_workload_metrics(temp_db):
    """Verify aggregation metrics for manager and collector dashboards."""
    df = pd.DataFrame({
        "loan_id": ["L1", "L2", "L3"],
        "probability": [0.70, 0.50, 0.10],
        "risk_bucket": ["High Risk", "Medium Risk", "Low Risk"],
        "warning_signs": [["W1"], ["W2"], ["W3"]],
        "collector_id": ["COL_104", "COL_104", "COL_104"],
        "loan_amount": [100000.0, 100000.0, 100000.0],
        "outstanding_amount": [50000.0, 40000.0, 10000.0],
        "emi_amount": [5000.0, 5000.0, 5000.0],
        "monthly_income": [30000.0, 30000.0, 30000.0],
        "days_past_due": [30, 15, 0],
        "suggested_action": ["A1", "A2", "A3"],
    })
    save_scored_loans(df, db_path=temp_db)

    # Mark one completed
    add_or_update_followup("L1", "COL_104", "Completed", "Done", "COL_104", db_path=temp_db)

    kpis = get_collector_kpis(collector_id="COL_104", db_path=temp_db)
    assert kpis["total_assigned"] == 3
    assert kpis["completed_count"] == 1
    assert kpis["pending_count"] == 2
    assert kpis["high_risk_count"] == 1
    assert kpis["total_outstanding"] == 100000.0
    assert kpis["completion_pct"] == 33.3
