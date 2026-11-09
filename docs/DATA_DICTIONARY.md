# LoanWatch Data Dictionary & Temporal Governance

## 1. Overview
The LoanWatch dataset is structured as a **loan-month panel** covering **5,000 synthetic loans** observed monthly over **36 chronological months** (`2023-01-01` to `2025-12-01`). Each record represents the point-in-time state of an active loan at a distinct monthly `observation_date` (the 1st of each calendar month).

This design reflects real-world Non-Banking Financial Company (NBFC) operations where active portfolios are scanned monthly to identify early deterioration signals before accounts cross into Non-Performing Asset (NPA) status (90+ Days Past Due).

---

## 2. Raw Schema Specification

| Column Name | Data Type | Units / Format | Allowed Values / Range | Inference Upload | Description |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `loan_id` | String (Categorical) | Alphanumeric (`LN_XXXXXX`) | Unique per borrower contract | **Mandatory** | Unique loan account identifier. |
| `observation_date` | Date | `YYYY-MM-DD` | `2023-01-01` to `2025-12-01` | **Mandatory** | The monthly snapshot date on which portfolio risk is evaluated. |
| `loan_amount` | Float / Numeric | Indian Rupees (₹) | ₹50,000 – ₹10,00,000 | **Mandatory** | Original principal sanctioned and disbursed. |
| `outstanding_amount` | Float / Numeric | Indian Rupees (₹) | ₹0 – ₹12,00,000 | **Mandatory** | Remaining unpaid principal + accrued penal interest at observation. |
| `emi_amount` | Float / Numeric | Indian Rupees (₹) | ₹1,500 – ₹95,000 | **Mandatory** | Scheduled monthly installment payable by borrower. |
| `monthly_income` | Float / Numeric | Indian Rupees (₹) | ₹25,000 – ₹2,50,000 | **Mandatory** | Borrower's verified gross monthly income at underwriting or update. |
| `loan_tenure` | Integer | Months | 12, 24, 36, 48, 60 | **Mandatory** | Original sanctioned duration of the credit facility. |
| `emi_due_date` | Date | `YYYY-MM-DD` | 5th of observation month | **Mandatory** | Due date for the current cycle's scheduled installment. |
| `emi_paid_date` | Date | `YYYY-MM-DD` / Null | Calendar date or Null | *Optional* | Actual date borrower cleared installment (may be prior to due date). |
| `days_late` | Integer | Days | 0 – 120 | **Mandatory** | Days elapsed between due date and payment date (0 if on/before due). |
| `bounced_emi_count` | Integer | Count | $\ge 0$ | **Mandatory** | Cumulative number of auto-debit (NACH/e-Mandate) bounces to date. |
| `previous_late_payments` | Integer | Count | $\ge 0$ | **Mandatory** | Cumulative number of past billing cycles delayed > 0 days. |
| `months_on_book` (MOB) | Integer | Months | 1 – 60 | **Mandatory** | Elapsed billing cycles since origination. |
| `days_past_due` (DPD) | Integer | Days | 0 – 89 (Active cohort) | **Mandatory** | Overdue days at `observation_date` (reflects partial payments & curing). |
| `collector_id` | String (Categorical) | Categorical (`COL_XXX`) | `COL_101` to `COL_108` | **Mandatory** | Collection officer assigned to the account. |
| `default_next_3_months` | Integer (Binary) | 0 or 1 | 0 (Non-default), 1 (Default) | *Optional* | **Supervised Target**: 1 if loan hits $\ge 90$ DPD within next 3 months. |

---

## 3. Strict Temporal Leakage Rules

> [!IMPORTANT]
> In credit risk and early warning systems, temporal data leakage is the single most common failure mode. A model trained with look-ahead bias appears artificially accurate in backtests but fails disastrously in production.

### Rule 1: Point-in-Time Feature Isolation
- Any feature calculated for a loan at `observation_date` $T$ **MUST ONLY** use transactions, payment states, and balances observed on or before $T$.
- Rolling aggregations (e.g. `bounced_in_last_3m`, `mean_days_late_6m`, `dpd_trend`) must use backward-looking windows ($[T - W, T]$) and strictly exclude any data from $t > T$.

### Rule 2: Purely Forward-Looking Target
- The target `default_next_3_months` is defined strictly over the future window $(T, T + 3\text{ months}]$.
- Target definition:
  $$\text{default\_next\_3\_months}_i(T) = \begin{cases} 1 & \text{if } \max_{t \in (T, T + 3\text{m}]} \text{days\_past\_due}_i(t) \ge 90 \\ 0 & \text{otherwise} \end{cases}$$
- For active inference uploads, `default_next_3_months` is **optional** and unknown; the system estimates its probability $\hat{p}$.

### Rule 3: Time-Based Splitting with Embargo Gaps
- Random cross-validation or row-based random train-test splitting causes data leakage because the same borrower observed at $T$ and $T+1$ would share overlapping 3-month forward outcome windows.
- All train/validation/test splits are strictly partitioned chronologically across the **36 observation months**:
  1. **Train Period**: Months 1 to 22 (22 months: `2023-01-01` to `2024-10-01`, ~61% of timeline).
  2. **Embargo Gap 1**: Exactly 3 months (Months 23 to 25: `2024-11-01` to `2025-01-01`) discarded from validation/training to completely eliminate forward outcome window overlap.
  3. **Validation Period**: Months 26 to 29 (4 months: `2025-02-01` to `2025-05-01`, ~11%).
  4. **Embargo Gap 2**: Exactly 3 months (Months 30 to 32: `2025-06-01` to `2025-08-01`) discarded.
  5. **Out-of-Time (OOT) Test Period**: Months 33 to 36 (4 months: `2025-09-01` to `2025-12-01`, ~11%).

---

## 4. Cohort Governance & Edge Case Policies

### Policy A: Active Non-Default Loan Cohort
- Early Warning Systems (EWS) are designed to identify deteriorating accounts *before* they become Non-Performing Assets.
- Accounts that are already at $\ge 90$ DPD on `observation_date` $T$ are **excluded** from training and active early-warning scoring. Once an account reaches NPA status, it moves to the legal/recovery workflow, not the early-warning prevention queue.

### Policy B: Loans with Limited History (< 3 Months on Book)
- Loans originated recently have `months_on_book` $< 3$.
- **Handling in Feature Engineering & Training**:
  - Rather than discarding new loans, rolling statistics (e.g. 3-month bounce count, average days late) are calculated over the available history.
  - A dedicated binary indicator `new_loan_flag = 1` (where $\text{MOB} < 3$) is engineered to alert the model and collection officers that seasoning is short.
  - Baseline static underwriting variables (`loan_amount`, `emi_to_income_ratio`, `loan_tenure`) carry the risk signal until seasoning occurs.

### Policy C: Early Payment Logic (`emi_paid_date` < `emi_due_date`)
- Borrowers who pay several days before the monthly due date have `days_late = 0`.
- This is completely valid payment behavior (pre-payment / advance clearing) and is treated as a healthy payment indicator by validation and feature engineering modules.
