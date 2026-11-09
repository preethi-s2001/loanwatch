"""
Unit tests for Tableau exports and dashboard guides (src/exports.py).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import TABLEAU_DIR
from src.exports import generate_all_tableau_exports


def test_tableau_exports_generation():
    """Verify that all 4 Tableau CSV files and the guide exist and contain clean data."""
    # Run export pipeline
    generate_all_tableau_exports()

    expected_files = [
        "portfolio_overview.csv",
        "risk_distribution.csv",
        "overdue_analysis.csv",
        "collector_performance.csv",
    ]

    for fname in expected_files:
        fpath = TABLEAU_DIR / fname
        assert fpath.exists(), f"Missing required Tableau export: {fname}"
        df = pd.read_csv(fpath)
        assert len(df) > 0, f"Tableau export {fname} should not be empty."

    # Verify Guide
    guide_path = TABLEAU_DIR / "DASHBOARD_GUIDE.md"
    assert guide_path.exists(), "Missing tableau/DASHBOARD_GUIDE.md"
    guide_text = guide_path.read_text(encoding="utf-8")
    assert "Portfolio Executive Overview" in guide_text
    assert "Risk Tier" in guide_text
    assert "Longitudinal Overdue" in guide_text
    assert "Collector Performance" in guide_text
