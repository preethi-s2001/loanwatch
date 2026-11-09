"""
Inference and Prediction Pipeline for LoanWatch NBFC Early Warning System.

Scoring pipeline:
1. Inputs cleaned loan panel or upload data.
2. Identifies the latest active observation snapshot per borrower.
3. Computes point-in-time features.
4. Generates calibrated probabilities of default within the next 3 months.
5. Assigns risk tier buckets (High, Medium, Low), respectful suggested actions,
   and top-3 plain-English warning signs.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    EXCLUDE_CURRENT_DEFAULT_THRESHOLD,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    RISK_TIERS,
)
from src.explain import generate_warning_signs
from src.features import compute_point_in_time_features

logger = logging.getLogger(__name__)


def score_active_loans(
    df_clean: pd.DataFrame,
    model_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Scores the latest active observation snapshot for every loan in df_clean.

    Returns:
    DataFrame with:
    - loan_id, observation_date, collector_id
    - loan_amount, outstanding_amount, emi_amount, monthly_income, days_past_due
    - probability: Calibrated probability in [0, 1]
    - risk_bucket: High Risk, Medium Risk, Low Risk
    - warning_signs: Top 3 plain-English warning signs (List[str])
    - suggested_action: Respectful outreach recommendation
    """
    m_path = model_path or MODEL_PATH
    f_path = features_path or FEATURE_COLUMNS_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at: {m_path}. Run training pipeline first.")
    if not f_path.exists():
        raise FileNotFoundError(f"Feature columns schema not found at: {f_path}.")

    # Load model and feature schema
    model = joblib.load(m_path)
    with open(f_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # 1. Compute point-in-time features across panel history
    df_featured = compute_point_in_time_features(df_clean)

    # 2. Select loans present at the LATEST observation date in the dataset
    # (Excludes historical snapshots from matured or defaulted accounts from earlier cycles)
    latest_obs_date = df_featured["observation_date"].max()
    df_latest = df_featured[df_featured["observation_date"] == latest_obs_date].copy()

    # 3. Filter active cohort (< 90 DPD) - exclude already defaulted loans
    df_active = df_latest[df_latest["days_past_due"] < EXCLUDE_CURRENT_DEFAULT_THRESHOLD].copy()

    if df_active.empty:
        logger.warning("No active accounts (< 90 DPD) found in the evaluation dataset.")
        return pd.DataFrame()

    # 4. Generate Predictions
    X = df_active[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]
    df_active["probability"] = [round(float(p), 4) for p in probabilities]

    # 5. Risk Bucketing & Respectful Actions
    def get_tier_info(prob: float):
        if prob >= RISK_THRESHOLD_HIGH:
            tier_cfg = RISK_TIERS["HIGH"]
        elif prob >= RISK_THRESHOLD_MEDIUM:
            tier_cfg = RISK_TIERS["MEDIUM"]
        else:
            tier_cfg = RISK_TIERS["LOW"]
        return tier_cfg["label"], tier_cfg["action"]

    tier_tuples = [get_tier_info(p) for p in df_active["probability"]]
    df_active["risk_bucket"] = [t[0] for t in tier_tuples]
    df_active["suggested_action"] = [t[1] for t in tier_tuples]

    # 6. Generate Top 3 Warning Signs per Account
    warning_signs_list = []
    for _, row in df_active.iterrows():
        signs = generate_warning_signs(row, top_k=3)
        warning_signs_list.append(signs)

    df_active["warning_signs"] = warning_signs_list

    # 7. Format clean display output
    display_columns = [
        "loan_id",
        "observation_date",
        "collector_id",
        "loan_amount",
        "outstanding_amount",
        "emi_amount",
        "monthly_income",
        "days_past_due",
        "days_late",
        "bounced_emi_count",
        "probability",
        "risk_bucket",
        "warning_signs",
        "suggested_action",
    ]

    # Include any remaining engineered features for deep-dives
    all_cols = display_columns + [c for c in df_active.columns if c not in display_columns]
    df_scored = df_active[all_cols].reset_index(drop=True)

    return df_scored
