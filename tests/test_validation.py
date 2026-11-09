"""
Unit tests for data validation module (src/validation.py).
Tests schema validation, error detection, duplicate removal, date logic,
and early payment handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DIRTY_SAMPLE_DATA_PATH,
    SAMPLE_DATA_PATH,
    UPLOAD_REQUIRED_COLUMNS,
)
from src.validation import validate_loan_data


def test_clean_sample_validation():
    """Verify that the clean sample dataset passes validation without errors."""
    df_clean = pd.read_csv(SAMPLE_DATA_PATH)
    report = validate_loan_data(df_clean)

    assert report.is_valid, f"Clean dataset should be valid. Errors: {report.errors}"
    assert len(report.errors) == 0, f"Expected 0 errors on clean data, got: {report.errors}"
    assert report.rows_retained == len(df_clean), "All clean rows should be retained."
    assert not report.cleaned_df.empty


def test_dirty_sample_validation_catches_anomalies():
    """Verify that validator detects injected errors, duplicates, and early payments."""
    df_dirty = pd.read_csv(DIRTY_SAMPLE_DATA_PATH)
    report = validate_loan_data(df_dirty)

    # Errors should be flagged for negative numbers, nulls, unparseable dates
    assert len(report.errors) > 0, "Dirty sample must produce validation errors."
    assert any("negative" in err.lower() or "violating rule" in err.lower() for err in report.errors), (
        "Should flag negative/impossible numeric values."
    )
    assert any("invalid or unparseable date" in err.lower() for err in report.errors), (
        "Should flag invalid date strings."
    )
    assert any("duplicate" in w.lower() for w in report.warnings), (
        "Should warn about and drop duplicates."
    )

    # Early payment case must NOT produce a fatal error (it is valid early settlement)
    early_warnings = [w for w in report.warnings if "prior to the scheduled EMI due date" in w]
    assert len(early_warnings) > 0, "Should generate informational warning for early payments."

    # Cleaned dataframe should exclude dirty rows
    assert report.rows_dropped > 0
    assert (report.cleaned_df["loan_amount"] > 0).all()
    assert (report.cleaned_df["outstanding_amount"] >= 0).all()
    assert (report.cleaned_df["monthly_income"] > 0).all()


def test_empty_dataframe_validation():
    """Verify handling of completely empty upload."""
    df_empty = pd.DataFrame()
    report = validate_loan_data(df_empty)

    assert not report.is_valid
    assert "empty" in report.errors[0].lower()


def test_missing_mandatory_columns():
    """Verify fatal error when mandatory columns are omitted."""
    df_incomplete = pd.DataFrame({
        "loan_id": ["LN_101"],
        "observation_date": ["2024-01-01"],
    })
    report = validate_loan_data(df_incomplete)

    assert not report.is_valid
    assert any("Missing mandatory columns" in err for err in report.errors)
