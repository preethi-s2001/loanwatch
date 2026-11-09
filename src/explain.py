"""
Explainability module for LoanWatch NBFC Early Warning System.

Generates human-readable, model-informed warning indicators in plain English.
Labels them strictly as 'warning signs' (NOT definitive 'causes').
Ensures respectful, non-accusatory language aligned with fair lending practices.
"""

from typing import List, Tuple
import pandas as pd


def generate_warning_signs(row: pd.Series, top_k: int = 3) -> List[str]:
    """
    Evaluates borrower point-in-time metrics and generates up to top_k plain-English warning signs.

    Warning signs are prioritized by risk severity and model feature importance:
    1. Delinquency state (days_past_due)
    2. Auto-debit bounce velocity (bounces_last_3m)
    3. Overdue days in current cycle (days_late)
    4. Delinquency trajectory (dpd_trend_3m)
    5. Chronic payment delays (late_payments_last_6m)
    6. Debt burden ratio (emi_to_income_ratio)
    7. Recent deterioration event
    8. Remaining principal exposure (outstanding_to_loan_ratio)
    9. Loan seasoning (new_loan_flag)
    """
    candidate_signs: List[Tuple[int, str]] = []

    # 1. Delinquency Position (Current DPD)
    dpd = int(row.get("days_past_due", 0))
    if dpd >= 60:
        candidate_signs.append((10, f"Account is {dpd} days past due (SMA-2 status)"))
    elif dpd >= 30:
        candidate_signs.append((9, f"Account is {dpd} days past due (SMA-1 status)"))
    elif dpd >= 10:
        candidate_signs.append((7, f"Account is {dpd} days past due (SMA-0 early delinquency)"))

    # 2. Auto-Debit Presentation Bounces
    bounces_3m = int(row.get("bounces_last_3m", 0))
    if bounces_3m >= 2:
        candidate_signs.append((10, f"{bounces_3m} auto-debit presentation bounces in last 3 months"))
    elif bounces_3m == 1:
        candidate_signs.append((8, "1 auto-debit presentation bounce in last 3 months"))

    # 3. Current Payment Delay
    days_late = int(row.get("days_late", 0))
    if days_late >= 20 and dpd < 30:
        candidate_signs.append((7, f"Current installment is overdue by {days_late} days"))

    # 4. Delinquency Trajectory (DPD Trend)
    trend = int(row.get("dpd_trend_3m", 0))
    if trend >= 20:
        candidate_signs.append((8, f"Delinquency trend: Overdue days rose by {trend} days over the last quarter"))

    # 5. Recent Deterioration Flag
    if int(row.get("recent_deterioration_flag", 0)) == 1:
        candidate_signs.append((8, "Recent deterioration: Auto-debit bounced or consecutive delinquency increase"))

    # 6. Chronic Payment Delays
    late_6m = int(row.get("late_payments_last_6m", 0))
    if late_6m >= 3:
        candidate_signs.append((6, f"Repeated payment friction: {late_6m} delayed installments in last 6 months"))
    elif late_6m >= 1:
        candidate_signs.append((5, f"{late_6m} delayed payment in last 6 months"))

    # 7. Debt Burden (FOIR)
    foir = float(row.get("emi_to_income_ratio", 0.0))
    foir_pct = round(foir * 100, 1)
    if foir >= 0.50:
        candidate_signs.append((6, f"Heavy debt burden: Scheduled EMI consumes {foir_pct}% of monthly income"))
    elif foir >= 0.40:
        candidate_signs.append((4, f"Elevated debt burden: Scheduled EMI consumes {foir_pct}% of monthly income"))

    # 8. Outstanding Principal Exposure
    out_ratio = float(row.get("outstanding_to_loan_ratio", 0.0))
    out_pct = round(out_ratio * 100, 1)
    if out_ratio >= 0.80 and dpd > 0:
        candidate_signs.append((3, f"High unamortized balance: {out_pct}% of original principal remains unpaid"))

    # 9. Short Seasoning Indicator
    if int(row.get("new_loan_flag", 0)) == 1:
        candidate_signs.append((2, "Short seasoning: Account on book for less than 3 months"))

    # Default fallback if account is fully healthy
    if not candidate_signs:
        candidate_signs.append((1, "Consistent on-time repayment history with 0 overdue days"))

    # Sort descending by priority weight, take top_k
    candidate_signs.sort(key=lambda x: x[0], reverse=True)
    selected_signs = [sign for _, sign in candidate_signs[:top_k]]

    return selected_signs
