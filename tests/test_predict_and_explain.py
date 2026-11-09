"""
Unit tests for Phase 5 Inference Pipeline (src/predict.py), Explainability (src/explain.py),
and Risk Bucket Thresholds.
"""

import pandas as pd
import pytest

from src.config import (
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    RISK_TIERS,
    SAMPLE_DATA_PATH,
)
from src.explain import generate_warning_signs
from src.predict import score_active_loans


def test_warning_signs_generation():
    """Verify explainability logic on high-risk, moderate-risk, and healthy loan profiles."""
    # Profile 1: Severe delinquency + multiple bounces
    high_risk_row = pd.Series({
        "days_past_due": 65,
        "bounces_last_3m": 2,
        "dpd_trend_3m": 35,
        "days_late": 65,
        "emi_to_income_ratio": 0.55,
        "recent_deterioration_flag": 1,
        "late_payments_last_6m": 3,
        "new_loan_flag": 0,
    })
    signs_high = generate_warning_signs(high_risk_row, top_k=3)
    assert len(signs_high) == 3
    assert any("SMA-2" in s for s in signs_high)
    assert any("bounce" in s.lower() for s in signs_high)

    # Profile 2: Clean on-time borrower
    clean_row = pd.Series({
        "days_past_due": 0,
        "bounces_last_3m": 0,
        "dpd_trend_3m": 0,
        "days_late": 0,
        "emi_to_income_ratio": 0.25,
        "recent_deterioration_flag": 0,
        "late_payments_last_6m": 0,
        "new_loan_flag": 0,
    })
    signs_clean = generate_warning_signs(clean_row, top_k=3)
    assert len(signs_clean) >= 1
    assert any("on-time" in s.lower() for s in signs_clean)


def test_risk_bucket_thresholds():
    """Verify exact cut-off behavior across High (>=0.60), Medium (0.30-0.60), Low (<0.30)."""
    assert RISK_THRESHOLD_HIGH == 0.60
    assert RISK_THRESHOLD_MEDIUM == 0.30

    # Test dummy scoring logic
    def get_tier(prob):
        if prob >= RISK_THRESHOLD_HIGH:
            return RISK_TIERS["HIGH"]["label"]
        elif prob >= RISK_THRESHOLD_MEDIUM:
            return RISK_TIERS["MEDIUM"]["label"]
        return RISK_TIERS["LOW"]["label"]

    assert get_tier(0.60) == "High Risk"
    assert get_tier(0.85) == "High Risk"
    assert get_tier(0.599) == "Medium Risk"
    assert get_tier(0.30) == "Medium Risk"
    assert get_tier(0.299) == "Low Risk"
    assert get_tier(0.02) == "Low Risk"


def test_score_active_loans_pipeline():
    """Verify that score_active_loans correctly processes a loan batch."""
    df_raw = pd.read_csv(SAMPLE_DATA_PATH).head(300)
    df_scored = score_active_loans(df_raw)

    assert not df_scored.empty
    assert "probability" in df_scored.columns
    assert "risk_bucket" in df_scored.columns
    assert "warning_signs" in df_scored.columns
    assert "suggested_action" in df_scored.columns

    # Verify probability bounds
    assert (df_scored["probability"] >= 0.0).all()
    assert (df_scored["probability"] <= 1.0).all()

    # Verify valid risk buckets
    allowed_buckets = {"High Risk", "Medium Risk", "Low Risk"}
    assert set(df_scored["risk_bucket"].unique()).issubset(allowed_buckets)

    # Verify respectful actions
    for action in df_scored["suggested_action"]:
        assert len(action) > 10
        assert "Consider" in action or "Initiate" in action or "Send" in action

    # Verify warning signs format
    for signs in df_scored["warning_signs"]:
        assert isinstance(signs, list)
        assert len(signs) >= 1
        assert len(signs) <= 3


def test_scored_loans_only_from_latest_observation_date():
    """
    Verify scoring bug fix:
    Only loans present at the latest observation date with DPD < 90 are scored.
    Fails if any scored loan lacks a snapshot at the latest observation date,
    or if any matured/defaulted loans from earlier cycles are included.
    """
    df_raw = pd.read_csv(SAMPLE_DATA_PATH)
    latest_date = df_raw["observation_date"].max()
    df_scored = score_active_loans(df_raw)

    assert not df_scored.empty

    # 1. Every single scored loan must have observation_date == latest_date
    assert (df_scored["observation_date"] == latest_date).all(), (
        f"Found scored loans with observation dates other than latest ({latest_date})"
    )

    # 2. Every scored loan must have days_past_due < 90 (active cohort, non-defaulted)
    assert (df_scored["days_past_due"] < 90).all(), (
        "Found scored loans with days_past_due >= 90 (already defaulted accounts should be excluded)"
    )

    # 3. Must match the exact active count present at latest date (1,223 loans)
    assert len(df_scored) == 1223, f"Expected 1,223 active loans, got {len(df_scored)}"
    assert df_scored["loan_id"].nunique() == 1223

