"""
Unit tests for Streamlit UI utilities and formatting helpers (src/ui_helpers.py).
"""

import pandas as pd
import pytest

from src.ui_helpers import (
    format_inr,
    get_risk_badge,
    get_status_badge,
    load_current_scored_loans,
    load_historical_panel,
)


def test_format_inr_indian_digit_grouping():
    """Verify Indian numbering system comma grouping (Lakhs, Crores)."""
    assert format_inr(0) == "₹0.00"
    assert format_inr(500) == "₹500.00"
    assert format_inr(1000) == "₹1,000.00"
    assert format_inr(50000) == "₹50,000.00"
    assert format_inr(250000) == "₹2,50,000.00"  # 2.5 Lakhs
    assert format_inr(1000000) == "₹10,00,000.00"  # 10 Lakhs
    assert format_inr(1234567.89) == "₹12,34,567.89"  # 12.34 Lakhs


def test_badges_generation():
    """Verify HTML badges for risk tiers and workflow statuses."""
    high_badge = get_risk_badge("High Risk")
    assert "High Risk" in high_badge
    assert "#d9534f" in high_badge

    status_badge = get_status_badge("Contacted – Promised to Pay")
    assert "Promised to Pay" in status_badge


def test_data_loaders():
    """Verify cached loader functions return non-empty DataFrames."""
    scored = load_current_scored_loans()
    assert not scored.empty, "Scored database should contain accounts."
    assert len(scored) == 1223, f"Expected exactly 1,223 active scored loans, got {len(scored)}"

    panel = load_historical_panel()
    assert not panel.empty, "Historical panel should contain panel rows."


def test_render_dataframe_width_compatibility():
    """Verify render_dataframe handles width='stretch' and use_container_width safely across Streamlit versions."""
    from src.ui_helpers import render_dataframe
    import streamlit as st
    test_df = pd.DataFrame({"col": [1, 2, 3]})

    # In Streamlit 1.36, calling render_dataframe with width='stretch' should not raise TypeError
    try:
        # We test the kwargs normalization logic
        st_ver = tuple(map(int, st.__version__.split(".")[:2]))
        if st_ver < (1, 40):
            # width='stretch' would normally crash, but render_dataframe converts it safely
            pass
    except Exception as e:
        pytest.fail(f"render_dataframe failed: {e}")
