"""
Synthetic Data Generator for LoanWatch NBFC Early Warning System.

Generates a realistic loan-month panel dataset simulating retail/MSME lending portfolios
in Indian Rupees (INR) across 36 observation months (5,000 loans).
Ensures strictly forward-looking target computation (no look-ahead leakage in features).
Target: default_next_3_months = 1 if loan reaches DPD >= 90 within next 3 months; else 0.
Cohort: Active loans (DPD < 90) on the observation date.
"""

import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from src.config import (
    COLLECTOR_IDS,
    DIRTY_SAMPLE_DATA_PATH,
    DPD_DEFAULT_THRESHOLD,
    EXCLUDE_CURRENT_DEFAULT_THRESHOLD,
    OUTCOME_WINDOW_MONTHS,
    RANDOM_SEED,
    RAW_COLUMNS,
    SAMPLE_DATA_PATH,
    TARGET_COLUMN,
    TOTAL_OBSERVATION_MONTHS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Calculate monthly EMI using standard reducing balance formula."""
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return round(principal / tenure_months, 2)
    factor = (1 + monthly_rate) ** tenure_months
    emi = principal * monthly_rate * factor / (factor - 1)
    return round(emi, 2)


def generate_synthetic_loans(
    n_loans: int = 5000,
    n_months: int = TOTAL_OBSERVATION_MONTHS,
    start_date: str = "2023-01-01",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic loan-month panel dataset with realistic, granular delinquency.
    Simulates n_months + OUTCOME_WINDOW_MONTHS (39 months total) to calculate
    ground-truth default_next_3_months forward targets for all 36 evaluation months.
    
    Delinquency variation:
    - Days past due varies across 0 to 89 days (partial payments, grace periods, delayed settlement).
    - Multi-signal interaction: loans with 0 DPD can default if burdened by high FOIR/shocks;
      loans with 30-70 DPD can cure or partially pay, preventing DPD alone from determining default.
    """
    np.random.seed(seed)
    base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    total_sim_months = n_months + OUTCOME_WINDOW_MONTHS

    # 1. Static Loan Origination Attributes
    loan_ids = [f"LN_{100000 + i}" for i in range(n_loans)]

    # Loan amounts: ₹50,000 to ₹10,00,000 (weighted towards ₹1L - ₹4L)
    loan_amounts = np.random.choice(
        [50000, 75000, 100000, 150000, 200000, 300000, 450000, 600000, 800000, 1000000],
        size=n_loans,
        p=[0.05, 0.08, 0.17, 0.20, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
    )

    # Loan tenures: 12 to 60 months
    tenures = np.random.choice([12, 24, 36, 48, 60], size=n_loans, p=[0.10, 0.35, 0.30, 0.15, 0.10])
    interest_rates = np.random.uniform(0.12, 0.22, size=n_loans)

    emis = np.array([
        calculate_emi(p, r, t) for p, r, t in zip(loan_amounts, interest_rates, tenures)
    ])

    # Latent credit risk propensity (Beta distribution, right-skewed)
    borrower_base_risk = np.random.beta(a=2.0, b=5.5, size=n_loans)

    # Monthly Incomes: realistic FOIR (Fixed Obligation to Income Ratio)
    incomes = []
    for emi, risk in zip(emis, borrower_base_risk):
        target_foir = np.clip(0.22 + 0.38 * risk + np.random.normal(0, 0.04), 0.15, 0.75)
        raw_income = emi / target_foir
        income = round(np.clip(raw_income, 25000, 250000) / 1000) * 1000
        incomes.append(income)
    monthly_incomes = np.array(incomes)

    # Staggered origination months (-18 to n_months - 6)
    origination_offsets = np.random.randint(-18, n_months - 6, size=n_loans)
    assigned_collectors = np.random.choice(COLLECTOR_IDS, size=n_loans)

    # 2. Simulate Monthly Trajectories
    panel_records = []

    for i in range(n_loans):
        l_id = loan_ids[i]
        l_amt = loan_amounts[i]
        l_emi = emis[i]
        l_tenure = tenures[i]
        l_income = monthly_incomes[i]
        l_collector = assigned_collectors[i]
        orig_m = origination_offsets[i]
        base_risk = borrower_base_risk[i]
        rate = interest_rates[i]

        curr_outstanding = float(l_amt)
        curr_dpd = 0
        bounced_count_cum = 0
        prev_late_count_cum = 0
        is_written_off = False

        # Persistent financial distress state (Markovian persistence)
        in_financial_distress = False

        for m in range(min(orig_m, 0), total_sim_months):
            mob = m - orig_m + 1
            if mob < 1:
                continue
            if mob > l_tenure or curr_outstanding <= 0 or is_written_off:
                break

            obs_date = base_date + relativedelta(months=m)
            due_date = obs_date.replace(day=5)

            # Distress transition
            foir = l_emi / l_income
            if not in_financial_distress:
                p_enter = 0.02 + 0.10 * base_risk + (0.06 if foir > 0.45 else 0.0)
                if np.random.rand() < p_enter:
                    in_financial_distress = True
            else:
                p_recover = 0.22 - 0.12 * base_risk
                if np.random.rand() < max(0.06, p_recover):
                    in_financial_distress = False

            effective_risk = np.clip(
                base_risk + (0.35 if in_financial_distress else 0.0) + (0.08 * (curr_dpd / 30.0)),
                0.02,
                0.98,
            )

            # Payment behavior with granular DPD and partial payment simulation
            if curr_dpd == 0:
                p_bounce = 0.05 + 0.32 * (effective_risk ** 1.3)
                p_delay = 0.08 + 0.20 * effective_risk

                draw = np.random.rand()
                if draw < p_bounce:
                    # Auto-debit bounced -> enters delinquency with granular DPD (5 to 34 days)
                    bounced_count_cum += 1
                    days_late = int(np.random.randint(15, 34))
                    curr_dpd = days_late
                    paid_date = due_date + relativedelta(days=days_late) if np.random.rand() < 0.35 else None
                elif draw < (p_bounce + p_delay):
                    # Delayed payment cleared within grace window (no bounce, DPD remains 0 at month-end)
                    prev_late_count_cum += 1
                    days_late = int(np.random.randint(2, 22))
                    curr_dpd = 0
                    paid_date = due_date + relativedelta(days=days_late)
                else:
                    # On-time or early payment
                    days_late = 0
                    curr_dpd = 0
                    early_days = int(np.random.choice([0, 1, 2, 3, 5], p=[0.65, 0.15, 0.10, 0.06, 0.04]))
                    paid_date = due_date - relativedelta(days=early_days)

            elif 1 <= curr_dpd < 30:
                # Early delinquency (SMA-0: 1 to 29 DPD)
                p_cure = 0.35 - 0.20 * effective_risk
                p_deteriorate = 0.45 + 0.30 * effective_risk

                draw = np.random.rand()
                if draw < p_cure:
                    prev_late_count_cum += 1
                    days_late = curr_dpd + int(np.random.randint(1, 10))
                    curr_dpd = 0
                    paid_date = due_date + relativedelta(days=min(days_late, 28))
                elif draw < (p_cure + p_deteriorate):
                    bounced_count_cum += 1
                    added_dpd = int(np.random.randint(25, 34))
                    curr_dpd = min(89, curr_dpd + added_dpd)
                    days_late = curr_dpd
                    paid_date = None
                else:
                    curr_dpd = int(np.random.randint(10, 28))
                    days_late = curr_dpd
                    paid_date = due_date + relativedelta(days=15)

            elif 30 <= curr_dpd < 60:
                # Moderate delinquency (SMA-1: 30 to 59 DPD)
                p_roll_forward = 0.42 + 0.32 * effective_risk
                p_cure_full = 0.18 - 0.10 * effective_risk
                p_partial = 0.22

                draw = np.random.rand()
                if draw < p_roll_forward:
                    bounced_count_cum += 1
                    added_dpd = int(np.random.randint(25, 33))
                    curr_dpd = min(89, curr_dpd + added_dpd)
                    days_late = curr_dpd
                    paid_date = None
                elif draw < (p_roll_forward + p_cure_full):
                    prev_late_count_cum += 1
                    days_late = curr_dpd
                    curr_dpd = 0
                    paid_date = due_date + relativedelta(days=20)
                elif draw < (p_roll_forward + p_cure_full + p_partial):
                    prev_late_count_cum += 1
                    curr_dpd = int(np.random.randint(8, 25))
                    days_late = curr_dpd
                    paid_date = due_date + relativedelta(days=22)
                else:
                    curr_dpd = int(np.clip(curr_dpd + np.random.randint(-5, 6), 30, 59))
                    days_late = curr_dpd
                    paid_date = None

            elif 60 <= curr_dpd < 90:
                # Severe delinquency (SMA-2: 60 to 89 DPD)
                p_default = 0.52 + 0.32 * effective_risk
                p_recovery = 0.20 - 0.10 * effective_risk

                draw = np.random.rand()
                if draw < p_default:
                    bounced_count_cum += 1
                    curr_dpd = int(np.random.randint(90, 115))
                    days_late = curr_dpd
                    paid_date = None
                elif draw < (p_default + p_recovery):
                    prev_late_count_cum += 1
                    curr_dpd = int(np.random.randint(15, 45))
                    days_late = curr_dpd
                    paid_date = due_date + relativedelta(days=25)
                else:
                    curr_dpd = int(np.clip(curr_dpd + np.random.randint(-4, 5), 60, 89))
                    days_late = curr_dpd
                    paid_date = None

            else:  # curr_dpd >= 90 (NPA)
                curr_dpd = min(curr_dpd + int(np.random.randint(25, 35)), 180)
                days_late = curr_dpd
                paid_date = None
                if curr_dpd >= 150:
                    is_written_off = True

            # Update loan balance
            principal_portion = max(0.0, l_emi - (curr_outstanding * (rate / 12.0)))
            if curr_dpd == 0:
                curr_outstanding = max(0.0, curr_outstanding - principal_portion)
            elif curr_dpd < 30:
                curr_outstanding = curr_outstanding + (l_emi * 0.005)
            elif curr_dpd < 60:
                curr_outstanding = curr_outstanding + (l_emi * 0.01)
            else:
                curr_outstanding = curr_outstanding + (l_emi * 0.02)

            if 0 <= m < total_sim_months:
                panel_records.append({
                    "loan_id": l_id,
                    "observation_date": obs_date.strftime("%Y-%m-%d"),
                    "month_index": m,
                    "loan_amount": float(l_amt),
                    "outstanding_amount": round(curr_outstanding, 2),
                    "emi_amount": float(l_emi),
                    "monthly_income": float(l_income),
                    "loan_tenure": int(l_tenure),
                    "emi_due_date": due_date.strftime("%Y-%m-%d"),
                    "emi_paid_date": paid_date.strftime("%Y-%m-%d") if paid_date else None,
                    "days_late": int(days_late),
                    "bounced_emi_count": int(bounced_count_cum),
                    "previous_late_payments": int(prev_late_count_cum),
                    "months_on_book": int(mob),
                    "days_past_due": int(curr_dpd),
                    "collector_id": l_collector,
                })

    df_panel = pd.DataFrame(panel_records)
    logger.info("Raw panel generated with %d records across %d loans.", len(df_panel), df_panel["loan_id"].nunique())

    # 3. Compute Strictly Forward-Looking Target (default_next_3_months)
    logger.info("Computing forward 3-month default target...")
    df_panel = df_panel.sort_values(["loan_id", "month_index"]).reset_index(drop=True)

    dpd_lookup = df_panel.set_index(["loan_id", "month_index"])["days_past_due"].to_dict()

    target_values = []
    for row in df_panel.itertuples():
        l_id = row.loan_id
        m = row.month_index
        future_dpds = [
            dpd_lookup.get((l_id, m + step), 0)
            for step in range(1, OUTCOME_WINDOW_MONTHS + 1)
        ]
        will_default = 1 if any(dpd >= DPD_DEFAULT_THRESHOLD for dpd in future_dpds) else 0
        target_values.append(will_default)

    df_panel[TARGET_COLUMN] = target_values

    # 4. Truncate back to the formal observation window (n_months: months 0 to n_months-1)
    df_formal = df_panel[df_panel["month_index"] < n_months].copy()

    # 5. Cohort Filtering: Exclude loans that are ALREADY in default (days_past_due >= 90) at observation
    df_active = df_formal[df_formal["days_past_due"] < EXCLUDE_CURRENT_DEFAULT_THRESHOLD].copy()

    df_clean = df_active[RAW_COLUMNS].copy()
    return df_clean


def generate_dirty_sample(df_clean: pd.DataFrame, n_dirty: int = 150, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate an anomaly-injected dirty sample dataset for validating upload error handling.
    Includes:
    - Paid before due date (valid real-world behavior to test date validation logic)
    - Negative amounts (loan_amount, outstanding_amount)
    - Missing values in mandatory columns
    - Unparseable date formats
    - Impossible tenure
    - Duplicate rows
    - Absence of target column (default_next_3_months is optional for uploads)
    """
    np.random.seed(seed)
    df_dirty = df_clean.sample(n=min(n_dirty, len(df_clean)), random_state=seed).copy()

    # 1. Negative amounts (Errors)
    idx_neg_loan = df_dirty.index[:15]
    df_dirty.loc[idx_neg_loan, "loan_amount"] = -50000.0

    idx_neg_out = df_dirty.index[15:30]
    df_dirty.loc[idx_neg_out, "outstanding_amount"] = -12500.0

    # 2. Missing values in mandatory columns (Errors)
    idx_null_income = df_dirty.index[30:45]
    df_dirty.loc[idx_null_income, "monthly_income"] = np.nan

    idx_null_emi = df_dirty.index[45:60]
    df_dirty.loc[idx_null_emi, "emi_amount"] = np.nan

    # 3. Unparseable date formats (Errors)
    idx_bad_date = df_dirty.index[60:75]
    df_dirty.loc[idx_bad_date, "observation_date"] = "INVALID_DATE_2024"

    # 4. Impossible values (Errors)
    idx_bad_tenure = df_dirty.index[75:90]
    df_dirty.loc[idx_bad_tenure, "loan_tenure"] = -12

    # 5. "Paid before due date" case (Valid real-world early payment, should be accepted / warned, not error)
    idx_early_pay = df_dirty.index[90:110]
    for idx in idx_early_pay:
        due_dt = pd.to_datetime(df_dirty.loc[idx, "emi_due_date"])
        early_pay_dt = due_dt - relativedelta(days=int(np.random.randint(2, 6)))
        df_dirty.loc[idx, "emi_paid_date"] = early_pay_dt.strftime("%Y-%m-%d")
        df_dirty.loc[idx, "days_late"] = 0

    # 6. Add duplicate rows
    duplicates = df_dirty.iloc[110:130].copy()
    df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

    # 7. Make default_next_3_months optional: drop or set to NaN for upload testing
    # In live uploads, users upload active loans without future target labels
    # Also strictly drop any temporary analysis columns like dpd_bucket
    df_dirty = df_dirty.drop(columns=[TARGET_COLUMN, "dpd_bucket"], errors="ignore")

    return df_dirty


def main():
    """CLI execution for generating synthetic datasets."""
    logger.info("Starting Phase 1 Synthetic Data Generation...")

    df_clean = generate_synthetic_loans(
        n_loans=5000,
        n_months=TOTAL_OBSERVATION_MONTHS,
        start_date="2023-01-01",
        seed=RANDOM_SEED
    )

    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(SAMPLE_DATA_PATH, index=False)
    logger.info("Saved clean synthetic dataset to: %s", SAMPLE_DATA_PATH)
    logger.info("Clean dataset shape: %s", df_clean.shape)

    overall_target_rate = df_clean[TARGET_COLUMN].mean()
    logger.info("Overall default target rate: %.2f%% (%d / %d)", overall_target_rate * 100, df_clean[TARGET_COLUMN].sum(), len(df_clean))

    # Log DPD breakdown on temporary Series without modifying df_clean in-place
    temp_dpd_bucket = pd.cut(
        df_clean["days_past_due"],
        bins=[-1, 0, 29, 59, 89],
        labels=["0 DPD", "1-29 DPD (SMA-0)", "30-59 DPD (SMA-1)", "60-89 DPD (SMA-2)"]
    )
    dpd_stats = df_clean.groupby(temp_dpd_bucket, observed=False)[TARGET_COLUMN].agg(
        total_rows="count",
        defaults="sum",
        default_rate=lambda s: round(s.mean() * 100, 2)
    )
    logger.info("\nDefault Rate by Days Past Due Bucket:\n%s", dpd_stats.to_string())

    df_dirty = generate_dirty_sample(df_clean)
    df_dirty.to_csv(DIRTY_SAMPLE_DATA_PATH, index=False)
    logger.info("Saved dirty dataset to: %s with %d rows", DIRTY_SAMPLE_DATA_PATH, len(df_dirty))


if __name__ == "__main__":
    main()
