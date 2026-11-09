# 🏦 LoanWatch – NBFC Early Warning System

> **Proactive Delinquency Detection, Multi-Signal Risk Scoring, and Prioritised Collection Workflows for Retail & MSME Credit Portfolios.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
In retail and MSME lending, Non-Banking Financial Companies (NBFCs) face substantial credit losses when overdue accounts cross the 90-day threshold into **Non-Performing Asset (NPA)** status. Traditional recovery operations rely on lagging indicators (e.g. contacting accounts only after 60 DPD), leading to collector alert fatigue and low cure rates.

**LoanWatch** is an institutional early-warning machine learning platform that analyzes **active, non-default loans** ($< 90$ DPD) to answer three operational questions:
1. **Which active accounts are likely to default within the next 3 months?**
2. **Why are they at risk?** (Plain-English, non-accusatory warning signs).
3. **Which accounts should collections officers follow up on first?** (Dual-key prioritization by risk probability and unamortized exposure).

---

## 🖥️ Application Screens & Visual Tour

| Screen | Primary User | Core Capability |
| :--- | :--- | :--- |
| **1. Manager Dashboard** | Risk Head / Collection Manager | Executive portfolio KPIs, risk tier distributions, 36-month longitudinal overdue trends, and collector workload allocation. |
| **2. Upload Data** | Operations / Data Analyst | Automated CSV schema verification, anomaly scrubbing (negative amounts, unparseable dates, duplicates), and one-click batch prediction. |
| **3. Risk Prediction** | Credit Risk Analyst | Calibrated default probability ($\hat{p}$) table, risk tier badges, top-3 warning indicators, and filtered CSV export. |
| **4. Collector Worklist** | Collection Executive | Prioritized daily queue with tie-breaking, borrower context, recommended outreach protocol, and SQLite interaction logging. |
| **5. Loan Details** | Credit Investigator | Single-loan deep-dive with borrower financial metrics, risk gauge, 36-month repayment trajectory chart, and chronological audit trail. |

*(Place screenshot images here when deploying to production: `docs/screenshots/dashboard.png`, `docs/screenshots/worklist.png`)*

---

## 🏗️ Architecture & Folder Structure

```text
loanwatch/
├── .streamlit/
│   └── config.toml             # Streamlit theme & headless server config
├── data/
│   ├── sample_loans.csv        # Clean synthetic 36-month panel (59,000+ records)
│   ├── dirty_sample_loans.csv  # Anomaly test file (negatives, nulls, unparseable dates)
│   ├── processed_features.csv  # Pre-computed point-in-time feature matrix
│   └── loanwatch.db            # SQLite database (loans_scored, followups)
├── docs/
│   ├── DATA_DICTIONARY.md      # Schema definitions, ranges, units & leakage rules
│   ├── MODEL_CARD.md           # Model architecture, OOT metrics, Brier calibration
│   ├── THRESHOLD_RATIONALE.md  # 4-step framework for tuning risk cut-offs
│   └── ETHICS.md               # Responsible AI, RBI compliance & anti-harassment
├── models/
│   ├── model.joblib            # Calibrated Random Forest pipeline artifact
│   ├── feature_columns.json    # Feature schema alignment specification
│   └── metrics.json            # Out-of-time evaluation metrics & calibration curves
├── notebooks/
│   └── 01_EDA.ipynb            # Fully executed EDA with embedded base64 charts
├── pages/
│   ├── 1_Manager_Dashboard.py  # Portfolio KPIs, risk distribution & workload
│   ├── 2_Upload_Data.py        # CSV validator, preview, and batch predictor
│   ├── 3_Risk_Prediction.py    # Probability scores, warning signs & filters
│   ├── 4_Collector_Worklist.py # Daily prioritized queue & follow-up logging
│   └── 5_Loan_Details.py       # Single-account trajectory & interaction history
├── scripts/                    # Standalone utility & execution scripts
│   ├── build_eda_notebook.py   # Generates executed EDA notebook with plots
│   ├── execute_notebook.py     # Headless Jupyter execution utility
│   ├── demo_phase5.py          # Phase 5 CLI scoring demonstration
│   └── inspect_phase1.py       # Initial synthetic distribution inspector
├── src/
│   ├── config.py               # Central paths, thresholds, and embargo settings
│   ├── data_generation.py      # Synthetic panel generator with roll-rate dynamics
│   ├── validation.py           # Schema checks, anomaly detection & deduplication
│   ├── features.py             # Point-in-time feature extraction (leakage-safe)
│   ├── train.py                # Chronological splitting, training & calibration
│   ├── evaluate.py             # PR-AUC, ROC-AUC, Brier score & confusion matrices
│   ├── predict.py              # Inference pipeline for active loans
│   ├── explain.py              # Plain-English warning signs generator
│   ├── worklist.py             # Dual-key queue prioritization engine
│   ├── db.py                   # SQLite persistence & audit log manager
│   ├── exports.py              # Pre-aggregated Tableau export builder
│   └── ui_helpers.py           # INR formatting (₹), badges, and cached loaders
├── tableau/
│   ├── portfolio_overview.csv  # Active portfolio loan-level export
│   ├── risk_distribution.csv   # Decile-aggregated risk tier export
│   ├── overdue_analysis.csv    # 36-month delinquency migration export
│   ├── collector_performance.csv# Caseload and completion rate export
│   └── DASHBOARD_GUIDE.md      # Step-by-step instructions for Tableau Public
├── tests/                      # Automated pytest suite (25 unit tests)
├── app.py                      # Main Streamlit web portal entry point
├── run_pipeline.py             # End-to-end one-command orchestrator
├── requirements.txt            # Pinned dependency manifest
├── runtime.txt                 # Python 3.11 declaration for Streamlit Cloud
├── pytest.ini                  # Test discovery configuration
└── .gitignore                  # Git ignore rules (.venv, cache, *.db)
```

---

## ⚡ Quick Start & Local Setup

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/loanwatch.git
cd loanwatch

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. One-Command Pipeline Execution
Run the full pipeline (generates synthetic data, extracts features, trains models, exports Tableau CSVs, and runs unit tests):
```bash
python run_pipeline.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Automated Testing
LoanWatch includes a comprehensive test suite covering schema integrity, zero look-ahead bias, model inference, database persistence, and UI helpers.

```bash
pytest -v
```
Expected output:
```text
tests/test_auth.py ....................... [ 18%]
tests/test_data_generation.py ........... [ 33%]
tests/test_db.py ......................... [ 45%]
tests/test_exports.py .................... [ 48%]
tests/test_features_leakage.py ........... [ 54%]
tests/test_model_pipeline.py ............. [ 63%]
tests/test_predict_and_explain.py ........ [ 75%]
tests/test_ui.py ......................... [ 87%]
tests/test_validation.py ................. [100%]
============================= 33 passed in 11.34s =============================
```

---

## 🔐 Authentication & Role-Based Access Control (RBAC)

LoanWatch includes session-based authentication backed by Streamlit secrets (`.streamlit/secrets.toml`), implementing PBKDF2-HMAC-SHA256 password hashing, rate-limiting lockout protection (5 failed attempts block for 30 seconds), and server-side collector data isolation.

### Role Hierarchy & Access Matrix

| Role | Permitted Pages | Data Scope | Follow-up Permissions |
| :--- | :--- | :--- | :--- |
| **`admin`** | All Pages | All Portfolios / Collectors | Read & Write |
| **`collection_manager`** | Manager Dashboard, Upload Data, Risk Prediction, Collector Worklist, Loan Details | All Portfolios / Collectors | Read & Write |
| **`risk_manager`** | Manager Dashboard, Upload Data, Risk Prediction, Collector Worklist, Loan Details | All Portfolios / Collectors | **Read-Only** (Updates Disabled) |
| **`business_head`** | **Manager Dashboard only** | All Portfolios | Read-Only |
| **`collector`** | Risk Prediction, Collector Worklist, Loan Details | **Assigned Collector ID only** | Read & Write (Own Accounts) |

### Setting Up Secrets Locally
1. Copy the provided template to initialize your local secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. The secrets file is gitignored (`.gitignore`) to ensure credentials are never committed.
3. Configure user profiles under `[auth.users.<user_key>]` with `name`, `role`, `collector_id`, and `password_hash`.

### Adding or Rotating Users
To compute a secure hash for a new user or password rotation without placing plaintext passwords in scripts:
```bash
python scripts/make_user_hash.py
```
- Enter and confirm the password interactively (input is masked and never echoed).
- Copy the generated `pbkdf2_sha256$200000$<salt>$<hash>` string directly into `.streamlit/secrets.toml` under `password_hash`.

### Streamlit Community Cloud Configuration
When deploying to Streamlit Community Cloud:
1. Open your deployed app on [share.streamlit.io](https://share.streamlit.io/).
2. Navigate to **App Settings** (gear icon) > **Secrets**.
3. Paste the contents of `.streamlit/secrets.toml` into the editor and click **Save**.

---

## ☁️ Deployment Guide: Streamlit Community Cloud

LoanWatch is designed for direct zero-configuration deployment to **Streamlit Community Cloud**:

1. **Python 3.11 Runtime:**
   A [`runtime.txt`](runtime.txt) containing `python-3.11` is committed to the project root, instructing Streamlit Community Cloud to provision an exact Python 3.11 runtime matching local development.

2. **Automated Database Auto-Seeding:**
   On cold starts or first runs where `data/loanwatch.db` is missing or unseeded, the application automatically triggers `ensure_database_seeded()` in [`src/ui_helpers.py`](src/ui_helpers.py). It scores the clean active cohort from `data/sample_loans.csv` and populates the SQLite database with 1,223 active accounts, ensuring immediate interactivity without manual file uploading.

3. **Ephemeral Storage Disclosure on Streamlit Cloud:**
   > [!IMPORTANT]
   > **Ephemeral Storage Notice:** Streamlit Community Cloud runs in stateless Linux containers. Any modifications to `data/loanwatch.db` (such as follow-up notes logged by collectors or new batch uploads) persist during the active browser session, but **are ephemeral across app restarts, idle sleep cycles, and container re-deployments**. On fresh container spins, the auto-seeder transparently rebuilds the database from the baseline sample data. In enterprise production, replace SQLite with an external PostgreSQL/MySQL connection using `st.connection("sql")`.

4. **Deploy on Streamlit Cloud:**
   - Push repository to GitHub (`main` branch).
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Click **"New app"**, select your repository, set **Main file path** to `app.py`.
   - Click **"Deploy!"**.

---

## ⚖️ Limitations & Ethical Disclaimers
1. **Synthetic Data Notice:** This software was trained and evaluated on synthetically simulated portfolio data. Reported performance metrics (Test PR-AUC 0.6913, Brier 0.0522; DPD-Only Baseline PR-AUC 0.6852) reflect synthetic delinquency transitions and do not represent actual performance on live bank books. Delinquency features (`days_past_due`, `days_late`, `dpd_trend_3m`) dominate the synthetic signal. Production use requires institutional retraining on audited core banking data.
2. **Authentication Security Level:** The authentication module is designed as an educational MVP for a synthetic-data proof-of-concept (PBKDF2 password hashing, local session state, 30s rate-limiting lockout). It does not provide enterprise-grade Multi-Factor Authentication (MFA), password reset workflows, or SIEM security audit logging. Institutional deployment requires integration with an enterprise IAM provider (e.g., Okta, Microsoft Entra ID, SAML/OIDC).
3. **Fair Lending & Human-in-the-Loop:** Risk scores are advisory decision-support tools. Automated adverse actions are strictly prohibited. All customer engagements must comply with the RBI Fair Practices Code for Lenders. See [`docs/ETHICS.md`](docs/ETHICS.md).

---

## 📄 License
This project is licensed under the MIT License. Developed as an applied data science and software engineering MVP.
