"""
Feature Engineering module for LoanWatch NBFC Early Warning System.

Computes point-in-time features strictly using data available on or before each
observation date T (zero look-ahead bias).
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ENGINEERED_FEATURE_COLUMNS,
    MIN_MONTHS_ON_BOOK_FEATURE,
    PROCESSED_DATA_PATH,
    SAMPLE_DATA_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def compute_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes point-in-time risk and behavioral features for a panel of active loans.

    Guarantees:
    - Chronological sorting per loan_id prior to any rolling aggregation.
    - Backward-only rolling windows (min_periods=1) to prevent look-ahead leakage.
    - Zero future knowledge: row at (loan_id, observation_date T) depends strictly
      on records at or before T.
    """
    df_feat = df.copy()

    # Ensure chronological sort per loan
    df_feat["observation_date"] = pd.to_datetime(df_feat["observation_date"])
    df_feat = df_feat.sort_values(["loan_id", "observation_date"]).reset_index(drop=True)

    # 1. Static and Point Ratios
    # EMI to Income Ratio (Fixed Obligation to Income Ratio - FOIR)
    df_feat["emi_to_income_ratio"] = (df_feat["emi_amount"] / df_feat["monthly_income"]).round(4)

    # Outstanding Balance to Original Sanctioned Principal Ratio
    df_feat["outstanding_to_loan_ratio"] = (df_feat["outstanding_amount"] / df_feat["loan_amount"]).round(4)

    # Tenure Remaining (Months)
    df_feat["tenure_remaining"] = np.maximum(0, df_feat["loan_tenure"] - df_feat["months_on_book"]).astype(int)

    # New Loan Indicator (Short seasoning flag: MOB < 3 months)
    df_feat["new_loan_flag"] = (df_feat["months_on_book"] < MIN_MONTHS_ON_BOOK_FEATURE).astype(int)

    # 2. Monthly Granular Behavioral Indicators
    grouped = df_feat.groupby("loan_id", sort=False)

    # Indicator for late payment in current cycle
    df_feat["_is_late"] = (df_feat["days_late"] > 0).astype(int)

    # Bounces that occurred specifically in the current cycle
    # Since bounced_emi_count is cumulative, diff() gives new bounces in this cycle
    df_feat["_cycle_bounces"] = (
        grouped["bounced_emi_count"]
        .diff()
        .fillna(df_feat["bounced_emi_count"])
        .clip(lower=0)
        .astype(int)
    )

    # Re-group on the Series directly for high-speed rolling calculations
    grouped_late = df_feat.groupby("loan_id", sort=False)["_is_late"]
    grouped_bounces = df_feat.groupby("loan_id", sort=False)["_cycle_bounces"]
    grouped_days_late = df_feat.groupby("loan_id", sort=False)["days_late"]
    grouped_dpd = df_feat.groupby("loan_id", sort=False)["days_past_due"]

    # 3. Rolling Backward-Looking Behavioral Features (Strictly t <= T)
    # Late payments in last 3 months and last 6 months
    df_feat["late_payments_last_3m"] = (
        grouped_late.rolling(3, min_periods=1).sum().reset_index(drop=True).astype(int)
    )
    df_feat["late_payments_last_6m"] = (
        grouped_late.rolling(6, min_periods=1).sum().reset_index(drop=True).astype(int)
    )

    # Auto-debit presentation bounces in last 3 months and 6 months
    df_feat["bounces_last_3m"] = (
        grouped_bounces.rolling(3, min_periods=1).sum().reset_index(drop=True).astype(int)
    )
    df_feat["bounces_last_6m"] = (
        grouped_bounces.rolling(6, min_periods=1).sum().reset_index(drop=True).astype(int)
    )

    # Days Late: Max and Mean over 3-month and 6-month windows
    df_feat["max_days_late_3m"] = (
        grouped_days_late.rolling(3, min_periods=1).max().reset_index(drop=True).astype(int)
    )
    df_feat["mean_days_late_3m"] = (
        grouped_days_late.rolling(3, min_periods=1).mean().reset_index(drop=True).round(2)
    )
    df_feat["max_days_late_6m"] = (
        grouped_days_late.rolling(6, min_periods=1).max().reset_index(drop=True).astype(int)
    )
    df_feat["mean_days_late_6m"] = (
        grouped_days_late.rolling(6, min_periods=1).mean().reset_index(drop=True).round(2)
    )

    # 4. DPD Momentum & Trajectory
    # Trend: Current DPD minus DPD from 3 months prior (or earliest available)
    dpd_3m_ago = (
        grouped_dpd.shift(2)
        .fillna(grouped_dpd.shift(1))
        .fillna(df_feat["days_past_due"])
    )
    df_feat["dpd_trend_3m"] = (df_feat["days_past_due"] - dpd_3m_ago).astype(int)

    # 5. Recent Deterioration Flag
    # 1 if DPD increased consecutively over past 2 cycles or if bounce occurred after clean history
    dpd_lag1 = grouped_dpd.shift(1).fillna(df_feat["days_past_due"])
    dpd_lag2 = grouped_dpd.shift(2).fillna(dpd_lag1)

    consecutive_dpd_rise = (df_feat["days_past_due"] > dpd_lag1) & (dpd_lag1 > dpd_lag2)
    recent_bounce = (df_feat["_cycle_bounces"] > 0) & (dpd_lag1 == 0)

    df_feat["recent_deterioration_flag"] = (consecutive_dpd_rise | recent_bounce).astype(int)

    # Drop temporary calculation columns
    df_feat = df_feat.drop(columns=["_is_late", "_cycle_bounces"])

    # Format observation_date back to string for consistency
    df_feat["observation_date"] = df_feat["observation_date"].dt.strftime("%Y-%m-%d")

    return df_feat


def prepare_features_for_modeling(
    df_clean: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the feature matrix X and target y for model training or evaluation.
    """
    cols = feature_cols or ENGINEERED_FEATURE_COLUMNS
    df_featured = compute_point_in_time_features(df_clean)

    X = df_featured[cols].copy()
    y = df_featured[TARGET_COLUMN].copy() if TARGET_COLUMN in df_featured.columns else pd.Series(dtype=int)

    return X, y


def main():
    """CLI execution for generating and saving engineered features dataset."""
    logger.info("Computing point-in-time features on clean dataset...")
    df_clean = pd.read_csv(SAMPLE_DATA_PATH)
    df_features = compute_point_in_time_features(df_clean)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Saved processed features to: %s (shape: %s)", PROCESSED_DATA_PATH, df_features.shape)


if __name__ == "__main__":
    main()
