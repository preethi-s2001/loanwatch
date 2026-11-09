"""
LoanWatch Configuration Module.
Defines constants, paths, thresholds, target criteria, and splitting logic.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Base Paths (Relative to project root for portability)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
TABLEAU_DIR = PROJECT_ROOT / "tableau"

# File Paths
SAMPLE_DATA_PATH = DATA_DIR / "sample_loans.csv"
DIRTY_SAMPLE_DATA_PATH = DATA_DIR / "dirty_sample_loans.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_features.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
DB_PATH = DATA_DIR / "loanwatch.db"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Time-Based Splitting & Embargo Configuration
# Outcome window is 3 months ahead; embargo prevents label overlap between splits
EMBARGO_MONTHS = 3
TOTAL_OBSERVATION_MONTHS = 36
TRAIN_MONTHS_COUNT = 22  # Months 0..21 (~61%)
VAL_MONTHS_COUNT = 4     # Months 25..28 (~11%)
TEST_MONTHS_COUNT = 4    # Months 32..35 (~11%)
# Embargo 1: Months 22..24 (3 months)
# Embargo 2: Months 29..31 (3 months)

# Target Definition:
# default_next_3_months = 1 if DPD reaches >= 90 within 3 months after observation_date; else 0
TARGET_COLUMN = "default_next_3_months"
DPD_DEFAULT_THRESHOLD = 90
OUTCOME_WINDOW_MONTHS = 3

# Cohort Filtering Rules
# Exclude loans already in default (DPD >= 90) on the observation date
EXCLUDE_CURRENT_DEFAULT_THRESHOLD = 90
# Minimum months on book required for rolling 3-month metrics without imputation flag
MIN_MONTHS_ON_BOOK_FEATURE = 3

# Risk Tier Thresholds (Tuned with risk committee / domain guidelines)
RISK_THRESHOLD_HIGH = 0.60
RISK_THRESHOLD_MEDIUM = 0.30

RISK_TIERS = {
    "HIGH": {
        "min_prob": RISK_THRESHOLD_HIGH,
        "max_prob": 1.0,
        "label": "High Risk",
        "action": "Consider a respectful in-person follow-up and restructuring check",
        "badge_color": "red",
    },
    "MEDIUM": {
        "min_prob": RISK_THRESHOLD_MEDIUM,
        "max_prob": RISK_THRESHOLD_HIGH,
        "label": "Medium Risk",
        "action": "Initiate a courteous telephonic check-in to confirm payment arrangement",
        "badge_color": "orange",
    },
    "LOW": {
        "min_prob": 0.0,
        "max_prob": RISK_THRESHOLD_MEDIUM,
        "label": "Low Risk",
        "action": "Send standard automated SMS/App payment reminder",
        "badge_color": "green",
    },
}

# Raw Dataset Columns (Full Panel with target)
RAW_COLUMNS = [
    "loan_id",
    "observation_date",
    "loan_amount",
    "outstanding_amount",
    "emi_amount",
    "monthly_income",
    "loan_tenure",
    "emi_due_date",
    "emi_paid_date",
    "days_late",
    "bounced_emi_count",
    "previous_late_payments",
    "months_on_book",
    "days_past_due",
    "collector_id",
    "default_next_3_months",
]

# Columns required for inference / user upload (default_next_3_months is optional for prediction)
UPLOAD_REQUIRED_COLUMNS = [
    "loan_id",
    "observation_date",
    "loan_amount",
    "outstanding_amount",
    "emi_amount",
    "monthly_income",
    "loan_tenure",
    "emi_due_date",
    "days_late",
    "bounced_emi_count",
    "previous_late_payments",
    "months_on_book",
    "days_past_due",
    "collector_id",
]
UPLOAD_OPTIONAL_COLUMNS = ["emi_paid_date", "default_next_3_months"]

# Allowed Follow-up Statuses
FOLLOWUP_STATUSES = [
    "Pending",
    "Contacted – No Answer",
    "Contacted – Promised to Pay",
    "Paid",
    "Escalated",
    "Completed",
]

# Collectors list for portfolio assignment
COLLECTOR_IDS = [
    "COL_101", "COL_102", "COL_103", "COL_104",
    "COL_105", "COL_106", "COL_107", "COL_108"
]

# Engineered Features list (Point-in-time temporal features)
ENGINEERED_FEATURE_COLUMNS = [
    "emi_to_income_ratio",
    "outstanding_to_loan_ratio",
    "tenure_remaining",
    "months_on_book",
    "days_past_due",
    "days_late",
    "late_payments_last_3m",
    "late_payments_last_6m",
    "bounces_last_3m",
    "bounces_last_6m",
    "max_days_late_3m",
    "mean_days_late_3m",
    "max_days_late_6m",
    "mean_days_late_6m",
    "dpd_trend_3m",
    "recent_deterioration_flag",
    "new_loan_flag",
]

# Baseline DPD-only feature
BASELINE_FEATURE_COLUMNS = ["days_past_due"]

# Complete feature set used for ML model training and inference
MODEL_FEATURE_COLUMNS = ENGINEERED_FEATURE_COLUMNS


def get_split_dates(unique_dates: List[pd.Timestamp]) -> Dict[str, List[pd.Timestamp]]:
    """
    Computes strict time-based split partitions with 3-month embargo gaps.
    """
    unique_dates = sorted(unique_dates)
    n = len(unique_dates)
    
    # 22 train, 3 embargo, 4 val, 3 embargo, 4 test = 36 total months
    train_end = min(TRAIN_MONTHS_COUNT, n - (2 * EMBARGO_MONTHS + VAL_MONTHS_COUNT + TEST_MONTHS_COUNT))
    val_start = train_end + EMBARGO_MONTHS
    val_end = val_start + VAL_MONTHS_COUNT
    test_start = val_end + EMBARGO_MONTHS
    
    return {
        "train": unique_dates[:train_end],
        "embargo_1": unique_dates[train_end:val_start],
        "validation": unique_dates[val_start:val_end],
        "embargo_2": unique_dates[val_end:test_start],
        "test": unique_dates[test_start:],
    }
