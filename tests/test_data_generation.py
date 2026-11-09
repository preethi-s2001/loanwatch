"""
Unit tests for Phase 1 Synthetic Data Generation, Schema Integrity, and Delinquency Realism.
"""

import pandas as pd
import pytest

from src.config import (
    DIRTY_SAMPLE_DATA_PATH,
    DPD_DEFAULT_THRESHOLD,
    EXCLUDE_CURRENT_DEFAULT_THRESHOLD,
    RAW_COLUMNS,
    SAMPLE_DATA_PATH,
    TARGET_COLUMN,
    UPLOAD_REQUIRED_COLUMNS,
    get_split_dates,
)


def test_clean_sample_exists_and_schema():
    assert SAMPLE_DATA_PATH.exists(), "Clean sample CSV must exist."
    df = pd.read_csv(SAMPLE_DATA_PATH)
    assert len(df) > 10000, "Dataset should have significant panel rows."
    for col in RAW_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_cohort_and_target_prevalence():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    # Confirm active cohort: no loans with DPD >= 90 at observation
    assert (df["days_past_due"] >= EXCLUDE_CURRENT_DEFAULT_THRESHOLD).sum() == 0, (
        "Active loans cohort must not contain loans already in default (DPD >= 90)."
    )
    # Prevalence between 8% and 15%
    prevalence = df[TARGET_COLUMN].mean()
    assert 0.08 <= prevalence <= 0.15, f"Target prevalence {prevalence:.2%} outside 8-15% target range."


def test_delinquency_granular_variation():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    # Ensure DPD is not just discrete {0, 30, 60}
    unique_dpd_values = df["days_past_due"].nunique()
    assert unique_dpd_values > 20, f"Expected continuous DPD variation, got only {unique_dpd_values} distinct values."

    # Verify monotonic relationship across DPD buckets without DPD being 100% deterministic
    df["dpd_bucket"] = pd.cut(df["days_past_due"], bins=[-1, 0, 29, 59, 89])
    rates = df.groupby("dpd_bucket", observed=False)[TARGET_COLUMN].mean()
    assert rates.iloc[0] < rates.iloc[1] < rates.iloc[2] < rates.iloc[3]
    # In 30-59 DPD bucket, not all loans default (some cure!)
    assert rates.iloc[2] < 0.90, "SMA-1 loans should have realistic curing rate."


def test_time_split_and_test_positive_count():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    unique_dates = pd.to_datetime(df["observation_date"]).drop_duplicates().sort_values().tolist()
    splits = get_split_dates(unique_dates)

    test_dates = [d.strftime("%Y-%m-%d") for d in splits["test"]]
    df_test = df[df["observation_date"].isin(test_dates)]

    test_positives = int(df_test[TARGET_COLUMN].sum())
    assert test_positives >= 300, f"Test split must have >= 300 positives. Found: {test_positives}"


def test_dirty_sample_contains_anomalies_and_early_payment():
    assert DIRTY_SAMPLE_DATA_PATH.exists(), "Dirty sample CSV must exist."
    df_dirty = pd.read_csv(DIRTY_SAMPLE_DATA_PATH)

    # 1. Target column is optional in upload format
    assert TARGET_COLUMN not in df_dirty.columns or df_dirty[TARGET_COLUMN].isna().any(), (
        "Upload dataset should treat default_next_3_months as optional."
    )

    # 2. Check all required upload columns exist
    for col in UPLOAD_REQUIRED_COLUMNS:
        assert col in df_dirty.columns, f"Upload data missing expected column: {col}"

    # 3. Check anomalies
    has_neg_loan = (df_dirty["loan_amount"] < 0).any()
    has_null_income = df_dirty["monthly_income"].isna().any()
    has_invalid_date = df_dirty["observation_date"].str.contains("INVALID").any()

    # 4. Check "Paid before due date" case exists (early payment)
    valid_dates = df_dirty.dropna(subset=["emi_paid_date", "emi_due_date"])
    valid_dates = valid_dates[~valid_dates["observation_date"].str.contains("INVALID", na=False)]
    early_payments = pd.to_datetime(valid_dates["emi_paid_date"]) < pd.to_datetime(valid_dates["emi_due_date"])

    assert has_neg_loan, "Dirty sample should contain negative loan amounts."
    assert has_null_income, "Dirty sample should contain missing income."
    assert has_invalid_date, "Dirty sample should contain invalid dates."
    assert early_payments.any(), "Dirty sample must include 'paid before due date' cases."
