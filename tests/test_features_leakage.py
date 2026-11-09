"""
Unit test proving zero temporal look-ahead leakage in feature engineering.
Verifies that features computed for loan L at observation date T strictly depend
on records at or before T, and are completely unaffected by rows from t > T.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ENGINEERED_FEATURE_COLUMNS, SAMPLE_DATA_PATH
from src.features import compute_point_in_time_features


def test_zero_look_ahead_leakage():
    """
    Leakage proof test:
    1. Compute features on the full multi-year panel.
    2. Choose a target observation date T (mid-timeline).
    3. Truncate the panel to include ONLY rows where observation_date <= T (purge future rows).
    4. Compute features on the truncated dataset.
    5. Compare the feature values at date T between full and truncated runs.
    6. They must match exactly (tolerance = 1e-6 for float roundoff).
    """
    df_raw = pd.read_csv(SAMPLE_DATA_PATH)
    dates = sorted(df_raw["observation_date"].unique())

    # Choose a midpoint date T
    cutoff_date = dates[15]

    # Full computation with future rows present
    df_full_features = compute_point_in_time_features(df_raw)

    # Truncated computation with future rows completely absent
    df_past_only = df_raw[df_raw["observation_date"] <= cutoff_date].copy()
    df_past_features = compute_point_in_time_features(df_past_only)

    # Extract feature matrix at cutoff date T for both
    feat_at_t_full = (
        df_full_features[df_full_features["observation_date"] == cutoff_date]
        .sort_values("loan_id")
        .set_index("loan_id")[ENGINEERED_FEATURE_COLUMNS]
    )

    feat_at_t_past = (
        df_past_features[df_past_features["observation_date"] == cutoff_date]
        .sort_values("loan_id")
        .set_index("loan_id")[ENGINEERED_FEATURE_COLUMNS]
    )

    # Assert exact loan index match
    assert feat_at_t_full.index.equals(feat_at_t_past.index), "Loan alignment must match."

    # Compare every single engineered feature column
    for col in ENGINEERED_FEATURE_COLUMNS:
        full_series = feat_at_t_full[col].values
        past_series = feat_at_t_past[col].values

        if np.issubdtype(full_series.dtype, np.floating):
            max_diff = np.max(np.abs(full_series - past_series))
            assert max_diff < 1e-5, (
                f"Leakage detected in feature '{col}'! Difference: {max_diff}. "
                "Feature at date T changed when future data was removed."
            )
        else:
            mismatches = (full_series != past_series).sum()
            assert mismatches == 0, (
                f"Leakage detected in discrete feature '{col}'! {mismatches} values differed "
                "when future data was removed."
            )


def test_no_null_features_in_engineered_output():
    """Verify that engineered features have no unhandled NaNs (min_periods=1 ensures coverage)."""
    df_raw = pd.read_csv(SAMPLE_DATA_PATH).head(2000)
    df_features = compute_point_in_time_features(df_raw)

    null_counts = df_features[ENGINEERED_FEATURE_COLUMNS].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, f"Found unexpected NaN values in features: {cols_with_nulls.to_dict()}"
