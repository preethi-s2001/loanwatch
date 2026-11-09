"""
Unit tests for Authentication, Role-Based Access Control (RBAC),
and Collector Data Isolation (src/auth.py).
Uses throwaway mock fixtures; never relies on production secrets.
"""

import time
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import streamlit as st

from src.auth import (
    CAN_WRITE_FOLLOWUPS_ROLES,
    PAGE_PERMISSIONS,
    ROLE_PERMISSIONS,
    can_write_followups,
    get_configured_users,
    get_current_user,
    hash_password,
    is_locked_out,
    is_page_allowed,
    login,
    logout,
    require_login,
    verify_password,
)
from src.config import DB_PATH
from src.db import add_or_update_followup, get_all_scored_loans_with_status, get_loan_followup_history


@pytest.fixture
def mock_throwaway_secrets():
    """Provides temporary throwaway credentials for testing; never uses production secrets."""
    pass_a = "throwaway_alpha_pass_123"
    pass_c = "throwaway_collector_pass_456"

    hash_a = hash_password(pass_a)
    hash_c = hash_password(pass_c)

    secrets_dict = {
        "auth": {
            "users": {
                "mock_admin": {
                    "name": "Mock Admin User",
                    "role": "admin",
                    "collector_id": "ALL",
                    "password_hash": hash_a,
                },
                "mock_collector": {
                    "name": "Mock Collector User",
                    "role": "collector",
                    "collector_id": "COL_101",
                    "password_hash": hash_c,
                },
            }
        }
    }
    with patch.object(st, "secrets", secrets_dict):
        yield {"pass_a": pass_a, "pass_c": pass_c}


def test_verify_password():
    """Verify PBKDF2 password verification under valid, invalid, empty, and malformed inputs."""
    pwd = "throwaway_test_password"
    valid_hash = hash_password(pwd)

    # 1. Correct password matches
    assert verify_password(pwd, valid_hash) is True

    # 2. Incorrect password fails
    assert verify_password("wrong_password", valid_hash) is False

    # 3. Empty password fails
    assert verify_password("", valid_hash) is False

    # 4. Malformed hashes return False without raising exceptions
    assert verify_password(pwd, "invalid_hash_string") is False
    assert verify_password(pwd, "pbkdf2_sha256$baditer$salt$hash") is False
    assert verify_password(pwd, "pbkdf2_sha256$-100$salt$hash") is False
    assert verify_password(pwd, "pbkdf2_sha256$1000$nothex$nothex") is False
    assert verify_password(pwd, "") is False
    assert verify_password(None, valid_hash) is False  # type: ignore
    assert verify_password(pwd, None) is False  # type: ignore


def test_login_and_lockout(mock_throwaway_secrets):
    """Verify login authentication flows, unknown users, and 5-failure 30s lockout."""
    # Reset session state
    st.session_state["auth_failed_attempts"] = 0
    st.session_state["auth_lockout_until"] = 0.0
    if "auth_user" in st.session_state:
        del st.session_state["auth_user"]

    good_pass = mock_throwaway_secrets["pass_a"]

    # 1. Unknown user fails
    assert login("nonexistent_user", "any_password") is False
    assert get_current_user() is None

    # 2. Known user with wrong password fails
    assert login("mock_admin", "incorrect_password") is False
    assert get_current_user() is None

    # 3. Known user with correct password succeeds
    assert login("mock_admin", good_pass) is True
    user = get_current_user()
    assert user is not None
    assert user["user_id"] == "mock_admin"
    assert user["role"] == "admin"
    assert user["collector_id"] == "ALL"

    # Test logout
    logout()
    assert get_current_user() is None

    # 4. Verify 5-failure lockout
    st.session_state["auth_failed_attempts"] = 0
    st.session_state["auth_lockout_until"] = 0.0

    for i in range(5):
        assert login("mock_admin", f"wrong_try_{i}") is False

    # Check that session is now locked out
    locked, remaining = is_locked_out()
    assert locked is True
    assert remaining > 0

    # Even with correct password, login must be blocked during lockout
    assert login("mock_admin", good_pass) is False


def test_role_to_page_permissions():
    """
    Verify strict role-based access control rules:
    - business_head cannot open anything except 1_Manager_Dashboard
    - risk_manager cannot write follow-ups
    - collector cannot open Manager Dashboard or Upload Data
    """
    # 1. Business Head: Manager Dashboard only
    assert is_page_allowed("business_head", "1_Manager_Dashboard") is True
    assert is_page_allowed("business_head", "2_Upload_Data") is False
    assert is_page_allowed("business_head", "3_Risk_Prediction") is False
    assert is_page_allowed("business_head", "4_Collector_Worklist") is False
    assert is_page_allowed("business_head", "5_Loan_Details") is False
    assert can_write_followups("business_head") is False

    # 2. Risk Manager: All pages allowed, but READ-ONLY follow-ups
    assert is_page_allowed("risk_manager", "1_Manager_Dashboard") is True
    assert is_page_allowed("risk_manager", "2_Upload_Data") is True
    assert is_page_allowed("risk_manager", "3_Risk_Prediction") is True
    assert is_page_allowed("risk_manager", "4_Collector_Worklist") is True
    assert is_page_allowed("risk_manager", "5_Loan_Details") is True
    assert can_write_followups("risk_manager") is False  # Cannot write follow-ups

    # 3. Collection Manager: All pages, can write follow-ups
    assert is_page_allowed("collection_manager", "1_Manager_Dashboard") is True
    assert is_page_allowed("collection_manager", "2_Upload_Data") is True
    assert is_page_allowed("collection_manager", "4_Collector_Worklist") is True
    assert can_write_followups("collection_manager") is True

    # 4. Admin: All pages, can write follow-ups
    for page in PAGE_PERMISSIONS:
        assert is_page_allowed("admin", page) is True
    assert can_write_followups("admin") is True

    # 5. Collector: Restricted pages, can write follow-ups
    assert is_page_allowed("collector", "1_Manager_Dashboard") is False
    assert is_page_allowed("collector", "2_Upload_Data") is False
    assert is_page_allowed("collector", "3_Risk_Prediction") is True
    assert is_page_allowed("collector", "4_Collector_Worklist") is True
    assert is_page_allowed("collector", "5_Loan_Details") is True
    assert can_write_followups("collector") is True


def test_collector_data_isolation():
    """
    Verify server-side collector data isolation:
    - Queries filtered by collector_id return ONLY that collector's rows.
    - Other collectors' loans are excluded from the retrieved dataset and CSV export.
    """
    df_all = get_all_scored_loans_with_status(db_path=DB_PATH)
    assert not df_all.empty

    # Pick a collector present in the data
    target_collector = "COL_101"
    df_collector = get_all_scored_loans_with_status(collector_id=target_collector, db_path=DB_PATH)

    assert not df_collector.empty
    # Every single row in the collector dataset must belong to that collector
    assert (df_collector["collector_id"] == target_collector).all()
    assert len(df_collector) < len(df_all)

    # Pick a loan belonging to another collector
    other_loans = df_all[df_all["collector_id"] != target_collector]
    assert not other_loans.empty
    foreign_loan_id = other_loans.iloc[0]["loan_id"]

    # Verify that the foreign loan is strictly absent from the collector's accounts
    assert foreign_loan_id not in df_collector["loan_id"].values

    # Verify CSV export from this filtered dataset contains only the collector's rows
    csv_text = df_collector.to_csv(index=False)
    df_reloaded = pd.read_csv(pd.io.common.StringIO(csv_text))
    assert (df_reloaded["collector_id"] == target_collector).all()


def test_updated_by_records_user_id(temp_db):
    """Verify that follow-up logging explicitly records the logged-in user ID into followups.updated_by."""
    from src.db import init_db, save_scored_loans

    test_loan_df = pd.DataFrame({
        "loan_id": ["LN_AUDIT_001"],
        "observation_date": ["2025-12-01"],
        "probability": [0.72],
        "risk_bucket": ["High Risk"],
        "warning_signs": [["High DPD"]],
        "collector_id": ["COL_101"],
        "loan_amount": [500000.0],
        "outstanding_amount": [400000.0],
        "emi_amount": [15000.0],
        "monthly_income": [35000.0],
        "days_past_due": [45],
        "suggested_action": ["Contact borrower"],
    })
    save_scored_loans(test_loan_df, db_path=temp_db)

    # Log follow-up with a specific user ID
    acting_user_id = "mock_collector_agent"
    add_or_update_followup(
        loan_id="LN_AUDIT_001",
        collector_id="COL_101",
        status="Contacted – Promised to Pay",
        notes="Spoke with borrower; payment promised by Friday.",
        updated_by=acting_user_id,
        db_path=temp_db,
    )

    history = get_loan_followup_history("LN_AUDIT_001", db_path=temp_db)
    assert not history.empty
    latest = history.iloc[0]
    assert latest["updated_by"] == acting_user_id
    assert latest["status"] == "Contacted – Promised to Pay"


def test_missing_secrets_safe_message():
    """Verify that when secrets are missing or unconfigured, the app returns None safely without raising stack traces."""
    with patch.object(st, "secrets", {}):
        users = get_configured_users()
        assert users is None

        # require_login should catch missing secrets and stop cleanly
        with patch.object(st, "error") as mock_error, patch.object(st, "stop") as mock_stop:
            mock_stop.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                require_login()
            mock_error.assert_called_once_with("Application is not configured")
