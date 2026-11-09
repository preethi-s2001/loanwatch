# LoanWatch Tableau Public Dashboard Implementation Guide

This guide provides step-by-step instructions for building four institutional-grade credit risk dashboards in **Tableau Public** or **Tableau Desktop** using the exported datasets in the `tableau/` directory.

---

## 1. Data Sources & Entity Relationship Model

| Export File | Level of Granularity | Primary Key | Foreign Keys / Join Keys |
| :--- | :--- | :--- | :--- |
| `portfolio_overview.csv` | Loan-level active snapshot | `loan_id` | `collector_id`, `observation_date` |
| `risk_distribution.csv` | Decile-aggregated risk tiers | `(risk_bucket, probability_decile)` | `risk_bucket` |
| `overdue_analysis.csv` | Month-by-month panel timeline | `observation_date` | — |
| `collector_performance.csv` | Executive-level caseworker summary | `collector_id` | `collector_id` |

### Recommended Data Model in Tableau:
- For independent dashboards, connect each sheet directly to its dedicated CSV.
- If creating a unified multi-table workbook:
  - Connect to `portfolio_overview.csv` as the root table.
  - Join `collector_performance.csv` on: `portfolio_overview.collector_id = collector_performance.collector_id`.

---

## 2. Dashboard 1: Portfolio Executive Overview

### Objective:
Give senior NBFC risk executives an immediate summary of active capital exposure, delinquency staging, and debt-burden distribution.

### Source: `tableau/portfolio_overview.csv`

### Calculated Fields to Create:
1. **`[Expected Loss Exposure]`**:
   ```tableau
   [outstanding_amount] * [probability] * 0.60
   ```
   *(Assumes 60% Loss Given Default - LGD)*
2. **`[FOIR Stress Band]`**:
   ```tableau
   IF [foir_pct] >= 50 THEN "High Burden (≥50%)"
   ELSEIF [foir_pct] >= 35 THEN "Moderate (35-50%)"
   ELSE "Healthy (<35%)"
   END
   ```
3. **`[Delinquency Status]`**:
   ```tableau
   IF [days_past_due] = 0 THEN "0 DPD (Current)"
   ELSEIF [days_past_due] < 30 THEN "1-29 DPD (SMA-0)"
   ELSEIF [days_past_due] < 60 THEN "30-59 DPD (SMA-1)"
   ELSE "60-89 DPD (SMA-2)"
   END
   ```

### Visual Components to Build:
1. **KPI Scorecard Strip (Top Row):**
   - **Card 1: Active Accounts:** `COUNTD([loan_id])`
   - **Card 2: Total Outstanding Capital:** `SUM([outstanding_amount])` (Formatted in Currency INR ₹)
   - **Card 3: High-Risk Capital at Risk:** `SUM(IF [risk_bucket] = "High Risk" THEN [outstanding_amount] END)`
   - **Card 4: Expected Portfolio Loss:** `SUM([Expected Loss Exposure])`
2. **Chart 1: Delinquency Staging Breakdown (Bar Chart):**
   - **Rows:** `[Delinquency Status]`
   - **Columns:** `SUM([outstanding_amount])`
   - **Color:** `[Delinquency Status]` (Palette: Green `#27ae60`, Blue `#2980b9`, Orange `#f39c12`, Red `#d9534f`)
3. **Chart 2: Debt-to-Income (FOIR) vs Outstanding Balance (Scatter Plot):**
   - **Columns:** `[foir_pct]` (Average)
   - **Rows:** `[outstanding_amount]` (Sum)
   - **Marks:** Circle
   - **Detail:** `[loan_id]`
   - **Color:** `[risk_bucket]`

---

## 3. Dashboard 2: Risk Tier & Default Probability Distribution

### Objective:
Visualize model calibration, risk tier concentrations, and the concentration of high-probability default accounts.

### Source: `tableau/risk_distribution.csv`

### Visual Components to Build:
1. **Chart 1: Risk Tier Concentration (Donut Chart):**
   - **Marks:** Pie
   - **Color:** `[risk_bucket]` (High Risk `#d9534f`, Medium Risk `#f39c12`, Low Risk `#27ae60`)
   - **Angle / Size:** `SUM([total_outstanding])`
   - **Label:** `SUM([account_count])` and `% of Total`
2. **Chart 2: Probability Decile Distribution (Histogram / Bar):**
   - **Columns:** `[probability_decile]`
   - **Rows:** `SUM([account_count])`
   - **Marks:** Bar
   - **Color:** `AVG([avg_probability])` (Color gradient: Light Yellow to Dark Red)
   - **Tooltip:** Include `SUM([total_outstanding])` and `AVG([avg_dpd])`
3. **Chart 3: Average Days Past Due by Decile (Line / Step):**
   - **Columns:** `[probability_decile]`
   - **Rows:** `AVG([avg_dpd])`
   - **Marks:** Line with data point circles

---

## 4. Dashboard 3: Longitudinal Overdue & Delinquency Migration Analysis

### Objective:
Track the 36-month timeline of loan seasoning, monitoring whether 30+ DPD accounts and forward defaults are expanding or stabilizing.

### Source: `tableau/overdue_analysis.csv`

### Visual Components to Build:
1. **Chart 1: Dual-Axis Overdue Trajectory (Line + Bar Combination):**
   - **Columns:** `[observation_date]` (Continuous Date)
   - **Rows (Axis 1):** `[sma1_count]` and `[sma2_count]` (Stacked Bars, color `#f39c12` and `#e74c3c`)
   - **Rows (Axis 2 - Dual Axis):** `[dpd_30_plus_rate]` (Line, color `#2c3e50`, width 2)
   - **Right-click Axis 2:** Check *Synchronize Axis*.
2. **Chart 2: Forward Default Rate vs Historical Overdue Rate:**
   - **Columns:** `[observation_date]`
   - **Rows:** `[forward_default_rate]` and `[dpd_30_plus_rate]`
   - **Marks:** Dual Line Graph comparing actual 3-month forward defaults against observed 30+ DPD accounts.
3. **Chart 3: Portfolio Outstanding Balance Growth:**
   - **Columns:** `[observation_date]`
   - **Rows:** `SUM([total_outstanding_portfolio])` (Area Chart, soft blue fill)

---

## 5. Dashboard 4: Collector Performance & Caseload Progress

### Objective:
Equip collection managers to track executive-level field allocations, promise-to-pay adherence, and completion progress.

### Source: `tableau/collector_performance.csv`

### Calculated Fields to Create:
1. **`[Caseload Completion Rate]`**:
   ```tableau
   [completed_followups] / [total_assigned]
   ```
2. **`[Pending Workload Flag]`**:
   ```tableau
   IF [pending_followups] > 350 THEN "High Backlog"
   ELSEIF [pending_followups] > 200 THEN "Normal"
   ELSE "Low Backlog"
   END
   ```

### Visual Components to Build:
1. **Chart 1: Caseload Status Breakdown per Collector (100% Stacked Horizontal Bar):**
   - **Rows:** `[collector_id]`
   - **Columns:** `SUM([completed_followups])`, `SUM([promised_to_pay])`, `SUM([pending_followups])`
   - **Quick Table Calculation:** *Percent of Total* (Compute using *Table Across*)
   - **Color:** Completed (Green `#2ecc71`), Promised to Pay (Blue `#3498db`), Pending (Grey `#95a5a6`)
2. **Chart 2: Collector High-Risk Exposure Managed (Treemap or Packed Bubble):**
   - **Marks:** Treemap
   - **Detail:** `[collector_id]`
   - **Size:** `[total_outstanding_managed]`
   - **Color:** `[high_risk_assigned]` (Reds gradient)
3. **Table: Collector Operational KPI Leaderboard:**
   - Columns:
     - Collector ID
     - Total Assigned
     - High Risk Assigned
     - Promised to Pay
     - Completed Cases
     - Completion % (Formatted as `0.0%`)
     - Total Outstanding (₹)

---

## 6. Dashboard Layout & Publishing Checklist
- **Screen Dimensions:** Set dashboard size to *Generic Desktop (1366 × 768)* or *Automatic*.
- **Interactivity / Actions:**
  - Add a Dashboard Filter Action: Clicking on a `collector_id` or `risk_bucket` filters all adjacent charts.
- **Publishing to Tableau Public:**
  1. Open Tableau Desktop / Tableau Public.
  2. Drag and drop CSV files from `loanwatch/tableau/`.
  3. Create an Extract (`.hyper`) before publishing.
  4. Select **Server > Tableau Public > Save to Tableau Public As...**
  5. Enter title: *"LoanWatch – NBFC Early Warning System"*.
