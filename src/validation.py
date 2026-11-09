"""
Validation module for LoanWatch NBFC Early Warning System.

Performs schema verification, dtype coercion, anomaly detection, date ordering logic,
and deduplication for loan-month panel data and active loan uploads.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    EXCLUDE_CURRENT_DEFAULT_THRESHOLD,
    TARGET_COLUMN,
    UPLOAD_OPTIONAL_COLUMNS,
    UPLOAD_REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured report returned by the data validator."""
    is_valid: bool
    total_rows_input: int
    rows_retained: int
    rows_dropped: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation {status}: Input {self.total_rows_input} rows -> "
            f"Retained {self.rows_retained} ({self.rows_dropped} dropped). "
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"
        )


def validate_loan_data(
    df_raw: pd.DataFrame,
    filter_npa_defaults: bool = True,
    strict_mode: bool = False,
) -> ValidationReport:
    """
    Validates and cleans a loan panel or inference upload DataFrame.

    Checks:
    - Presence of mandatory columns.
    - Dtype conversions & string stripping.
    - Identification and removal of exact duplicates.
    - Date format validity and calendar logic.
    - Detection of impossible/negative numbers (loan_amount, EMI, income, tenure).
    - Handling of early payments (paid before due date is marked valid).
    - Separation of critical fatal errors vs actionable operational warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    total_rows = len(df_raw)
    if total_rows == 0:
        return ValidationReport(
            is_valid=False,
            total_rows_input=0,
            rows_retained=0,
            rows_dropped=0,
            errors=["Uploaded file is empty (0 rows)."],
            warnings=[],
            cleaned_df=pd.DataFrame(),
        )

    df = df_raw.copy()

    # 1. Mandatory Column Check
    missing_cols = [col for col in UPLOAD_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Missing mandatory columns: {', '.join(missing_cols)}"
        errors.append(msg)
        return ValidationReport(
            is_valid=False,
            total_rows_input=total_rows,
            rows_retained=0,
            rows_dropped=total_rows,
            errors=errors,
            warnings=warnings,
            cleaned_df=pd.DataFrame(),
        )

    # 2. Check for Duplicates
    dup_mask = df.duplicated(subset=["loan_id", "observation_date"], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        warnings.append(f"Detected {dup_count} duplicate loan-observation records. Dropping duplicates.")
        df = df[~dup_mask].copy()

    # 3. Date Coercion and Integrity
    date_cols = ["observation_date", "emi_due_date"]
    if "emi_paid_date" in df.columns:
        date_cols.append("emi_paid_date")

    invalid_date_mask = pd.Series(False, index=df.index)
    for col in ["observation_date", "emi_due_date"]:
        coerced = pd.to_datetime(df[col], errors="coerce")
        nat_count = int(coerced.isna().sum())
        if nat_count > 0:
            errors.append(f"Column '{col}' has {nat_count} invalid or unparseable date values.")
            invalid_date_mask = invalid_date_mask | coerced.isna()
        df[col] = coerced

    if invalid_date_mask.any():
        df = df[~invalid_date_mask].copy()

    # Parse optional emi_paid_date
    if "emi_paid_date" in df.columns:
        df["emi_paid_date"] = pd.to_datetime(df["emi_paid_date"], errors="coerce")
        # Check for paid before due date (valid early clearance)
        early_mask = (df["emi_paid_date"] < df["emi_due_date"]) & df["emi_paid_date"].notna()
        early_count = int(early_mask.sum())
        if early_count > 0:
            warnings.append(
                f"Informational: {early_count} payments were received prior to the scheduled EMI due date (advance clearance)."
            )

    # 4. Numeric Coercion and Range Validation
    numeric_specs = {
        "loan_amount": ("positive", 1000.0, 100000000.0),
        "outstanding_amount": ("non_negative", 0.0, 150000000.0),
        "emi_amount": ("positive", 100.0, 5000000.0),
        "monthly_income": ("positive", 5000.0, 50000000.0),
        "loan_tenure": ("positive_int", 1, 360),
        "days_late": ("non_negative_int", 0, 365),
        "bounced_emi_count": ("non_negative_int", 0, 100),
        "previous_late_payments": ("non_negative_int", 0, 100),
        "months_on_book": ("positive_int", 1, 360),
        "days_past_due": ("non_negative_int", 0, 365),
    }

    invalid_numeric_mask = pd.Series(False, index=df.index)

    for col, (rule, min_val, max_val) in numeric_specs.items():
        if col in df.columns:
            # Coerce to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} non-numeric or missing values.")
                invalid_numeric_mask = invalid_numeric_mask | df[col].isna()

            # Range checks
            out_of_bounds = (df[col] < min_val) | (df[col] > max_val)
            oob_count = int(out_of_bounds.sum())
            if oob_count > 0:
                errors.append(
                    f"Column '{col}' has {oob_count} values violating rule '{rule}' "
                    f"(expected range [{min_val}, {max_val}])."
                )
                invalid_numeric_mask = invalid_numeric_mask | out_of_bounds

    if invalid_numeric_mask.any():
        df = df[~invalid_numeric_mask].copy()

    # 5. Cohort Filtering: Exclude loans already in default (DPD >= 90)
    if filter_npa_defaults and "days_past_due" in df.columns:
        npa_mask = df["days_past_due"] >= EXCLUDE_CURRENT_DEFAULT_THRESHOLD
        npa_count = int(npa_mask.sum())
        if npa_count > 0:
            warnings.append(
                f"Excluded {npa_count} accounts already at or above 90 DPD (NPA cohort is routed to Legal/Recovery)."
            )
            df = df[~npa_mask].copy()

    # 6. Target column handling (Optional for uploads)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        # Ensure binary 0 or 1 if present
        valid_target = df[TARGET_COLUMN].isin([0, 1]) | df[TARGET_COLUMN].isna()
        if not valid_target.all():
            errors.append(f"Column '{TARGET_COLUMN}' contains non-binary values.")
            df = df[valid_target].copy()

    # 7. Helper Indicators for Downstream Processing
    df["is_payment_delayed"] = (df["days_late"] > 0).astype(int)
    df["has_bounced"] = (df["bounced_emi_count"] > 0).astype(int)

    # Determine validity
    retained_rows = len(df)
    dropped_rows = total_rows - retained_rows
    is_valid = len(errors) == 0 if strict_mode else retained_rows > 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows_input=total_rows,
        rows_retained=retained_rows,
        rows_dropped=dropped_rows,
        errors=errors,
        warnings=warnings,
        cleaned_df=df.reset_index(drop=True),
    )
