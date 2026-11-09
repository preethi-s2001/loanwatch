# LoanWatch Model Card: Active Loan Early Warning System

## 1. Model Details
- **Model Name:** LoanWatch Early Warning Default Predictor (`Random_Forest`)
- **Model Version:** 2.0.0
- **Model Architecture:** Depth-Limited Ensemble Decision Forest (100 trees, `max_depth=8`, `min_samples_leaf=25`)
- **Selection Criterion:** Programmatic selection strictly on **Validation set metrics** (Primary: PR-AUC; Secondary: Brier Score)
- **Developer:** LoanWatch Risk Analytics & Data Science Team
- **Release Date:** November 2026
- **License:** Open Academic / Student MVP

---

## 2. Intended Use & Target Users
- **Primary Purpose:** Identify active Non-Banking Financial Company (NBFC) retail/MSME loans at heightened probability of entering 90+ Days Past Due (DPD) default within the subsequent 3 months.
- **Intended Users:**
  1. **Collection Executives:** To view a prioritized daily worklist ranked by risk probability and outstanding balance, rather than simple lagging DPD.
  2. **Collection Managers:** To balance collector caseloads and monitor account curing trends.
  3. **Risk Managers & Credit Committees:** To monitor macroeconomic deterioration signals, migration across delinquency tiers, and portfolio loss exposure.
- **Out-of-Scope Uses:**
  - Automated punitive enforcement (e.g. automatic repossession, automated legal filing).
  - Underwriting rejection of fresh applicant credit files (this model is trained on active repayment behavior, not origination bureau scores alone).

---

## 3. Supervised Target Definition
$$\text{default\_next\_3\_months}_i(T) = \begin{cases} 1 & \text{if } \max_{t \in (T, T + 3\text{m}]} \text{days\_past\_due}_i(t) \ge 90 \\ 0 & \text{otherwise} \end{cases}$$
- **Window:** Evaluates delinquency events strictly occurring within the 3 calendar months *after* the observation date $T$.
- **Active Cohort Filter:** Excludes loans already at $\ge 90$ DPD at date $T$ (Non-Performing Assets are routed to legal/recovery, not early warning).

---

## 4. Temporal Split Strategy & Embargo Governance
Panel data cannot be randomly partitioned by row. Doing so causes temporal leakage because rows for the same borrower at months $T$ and $T+1$ share overlapping 3-month forward outcome windows.

A strict chronological split is implemented across 36 observation months:
- **Train Period (Months 1–22):** `2023-01-01` to `2024-10-01` (36,748 rows, 4,615 defaults, 12.56% rate).
- **Embargo Gap 1 (`EMBARGO_MONTHS = 3`):** `2024-11-01` to `2025-01-01` (5,815 rows purged to eliminate outcome overlap between Train and Validation).
- **Validation Period (Months 26–29):** `2025-02-01` to `2025-05-01` (7,228 rows, 876 defaults, 12.12% rate; used for model selection and calibration).
- **Embargo Gap 2 (`EMBARGO_MONTHS = 3`):** `2025-06-01` to `2025-08-01` (4,738 rows purged to isolate Test).
- **Out-of-Time (OOT) Test Period (Months 33–36):** `2025-09-01` to `2025-12-01` (**5,432 rows, 613 defaults**, 11.28% rate; strictly future holdout).

---

## 5. Programmatic Model Selection on Validation Partition

Models were trained without artificial class-weight distortions for fair calibration comparison. Candidate models were ranked programmatically on the **Validation partition** using **PR-AUC as the primary metric** and **Brier Score as the secondary metric**, evaluated at the operational alert threshold $\tau = 0.30$ (linked to the Medium Risk tier boundary):

| Candidate Architecture | Validation PR-AUC | Validation Brier | Validation ROC-AUC | Validation Recall | Validation F1 | Selection Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DPD-Only Baseline** | 0.6850 | 0.0598 | 0.9320 | 67.58% | 0.6527 | *Benchmark (Single-Feature)* |
| **DPD-Only Calibrated** | 0.6850 | 0.0598 | 0.9320 | 67.58% | 0.6527 | *Calibrated Benchmark* |
| **Logistic Regression** | 0.6739 | 0.0597 | 0.9324 | 71.69% | 0.6528 | Linear candidate |
| **Logistic Regression (Calibrated)** | 0.6739 | 0.0596 | 0.9324 | 70.32% | 0.6477 | Calibrated linear |
| **Decision Tree (depth=5)** | 0.6665 | 0.0559 | 0.9352 | 79.22% | 0.6692 | Rule-based shallow tree |
| **Decision Tree (Calibrated)** | 0.6665 | 0.0582 | 0.9352 | 69.06% | 0.6594 | Calibrated tree |
| **Random Forest** ★ | **0.6912** | **0.0559** | **0.9384** | **77.74%** | **0.6696** | **Selected Winner (Highest Val PR-AUC & Lowest Brier)** |
| **Random Forest (Calibrated)** | 0.6912 | 0.0580 | 0.9384 | 69.52% | 0.6638 | Calibrated ensemble |

★ **Programmatically Selected Production Model** based on validation performance.

---

## 6. Performance Evaluation on Out-of-Time Test Set

The selected model and candidate benchmarks were evaluated **once on the held-out Out-of-Time Test set** at the operational decision threshold $\tau = 0.30$:

| Model Architecture | Test PR-AUC | Test ROC-AUC | Test Brier Score | Test Precision | Test Recall | Test F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DPD-Only Baseline** (Benchmark) | 0.6852 | 0.9403 | 0.0570 | 59.49% | 71.62% | 0.6499 | Evaluates `days_past_due` alone |
| **Logistic Regression** | 0.6933 | 0.9422 | 0.0560 | 58.74% | 74.39% | 0.6459 | Scaled multivariable |
| **Decision Tree** | 0.6775 | 0.9487 | 0.0516 | 55.00% | 80.75% | 0.6543 | Shallow tree |
| **Random Forest** ★ (Final Model) | **0.6913** | **0.9433** | **0.0522** | **56.73%** | **79.12%** | **0.6608** | **Final Model: High recall (79.1%) & low Brier (0.0522)** |
| **Random Forest (Calibrated)** | 0.6913 | 0.9433 | 0.0539 | 61.78% | 71.45% | 0.6626 | Calibrated probabilities |

### Key Observation: DPD Dominance in Synthetic Data
> [!IMPORTANT]
> **Signal Dominance Disclosure:** In this synthetic dataset, **Days Past Due (`days_past_due`) and immediate delinquency momentum (`days_late`, `dpd_trend_3m`) dominate the predictive signal**.
> - The single-variable **DPD-Only Baseline** already achieves a strong Test PR-AUC of **0.6852** and Test ROC-AUC of **0.9403**.
> - The full **Random Forest model** achieves a Test PR-AUC of **0.6913**, providing an incremental lift of **+0.0061 (+0.89%)** over the DPD baseline.
> - This incremental gain is driven primarily by **short-term auto-debit bounce patterns (`bounces_last_3m`)** and peak delays (`max_days_late_3m`).
> - Static underwriting attributes like FOIR (`emi_to_income_ratio`) contribute $< 1\%$ to feature importance in this panel; borrowers default primarily due to cash-flow payment breaks rather than initial leverage ratios.

A visual reliability diagram comparing the calibration curves against perfect calibration ($y = x$) is persisted at [`models/calibration_curve.png`](file:///models/calibration_curve.png).

---

## 7. Feature Importance (Top Predictors)

Feature importances extracted directly from the fitted Random Forest ensemble:

1. `days_past_due` (26.23%): Current delinquency bucket / days overdue.
2. `days_late` (19.89%): Days overdue in the current billing cycle.
3. `dpd_trend_3m` (12.31%): 90-day trajectory of delinquency progression.
4. `max_days_late_3m` (9.86%): Peak lateness over the preceding quarter.
5. `bounces_last_3m` (8.58%): Auto-debit / NACH presentation bounces in last 3 months.
6. `mean_days_late_3m` (7.02%): Average delay over the past quarter.
7. `max_days_late_6m` (5.64%): 6-month historical peak delinquency.
8. `bounces_last_6m` (2.04%): Chronic auto-debit presentation bounce frequency.
9. `mean_days_late_6m` (2.04%): 6-month average payment lag.
10. `recent_deterioration_flag` (1.92%): Binary flag indicating jump in DPD.
11. `tenure_remaining` (1.13%): Unamortized loan duration remaining.
12. `outstanding_to_loan_ratio` (0.96%): Fraction of principal unpaid.
13. `emi_to_income_ratio` (0.84%): Underwriting FOIR leverage ratio.
14. `late_payments_last_3m` (0.64%): Count of delinquent installments in last 3m.
15. `late_payments_last_6m` (0.45%): Count of delinquent installments in last 6m.
16. `months_on_book` (0.43%): Loan seasoning duration.
17. `new_loan_flag` (0.01%): Early-stage seasoning indicator.

---

## 8. Limitations & Responsible Governance
1. **Synthetic Data Disclaimer:** This model is trained and evaluated entirely on synthetically generated panel trajectories. Synthetic data patterns do **NOT** guarantee equivalent accuracy on real NBFC credit books. Production deployment requires retraining on audited Core Banking System (CBS) / Loan Management System (LMS) data.
2. **Prohibited Demographics:** Demographics and protected classes (gender, religion, caste, marital status, residential pincode proxy) are strictly excluded from the training schema.
3. **Advisory Human-in-the-Loop:** Predictions $\hat{p} \in [0, 1]$ represent advisory statistical alerts for collection managers and outreach officers, **never legal proof of default**. All borrower interactions must strictly follow the RBI Fair Practices Code: polite dialogue, restricted contact hours (08:00–19:00), and absolute prohibition of automated harassment.
