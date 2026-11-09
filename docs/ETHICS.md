# LoanWatch Ethical Governance, Responsible AI & Fair Lending Principles

## 1. Executive Charter & Purpose
LoanWatch is an **Early Warning System (EWS)** designed to support credit risk and collections operations within Non-Banking Financial Companies (NBFCs). Its goal is to provide predictive intelligence to human officers to help borrowers navigate financial distress **before** an active account crosses into 90+ Days Past Due (DPD) Non-Performing Asset (NPA) status.

Because machine learning algorithms in credit collections interface directly with individuals experiencing financial vulnerability, strict ethical safeguards must govern every stage of model development, outreach, and operational enforcement.

---

## 2. Core Ethical Principles

### Principle 1: Advisory Decision-Support (Strict Human-in-the-Loop)
- The model outputs an advisory calibrated probability of default ($\hat{p} \in [0, 1]$), **NOT a deterministic judgment or proof of default**.
- **No Automated Enforcement:** The system must never trigger automated punitive measures (such as automatic vehicle repossession, collateral seizure, bank account freezing, or automated legal notice filing) based on a risk score.
- Every outreach action requires human judgment and discretion by a qualified collection executive.

### Principle 2: Prohibited Features & Anti-Discrimination
- The model strictly excludes protected demographic, cultural, and personal characteristics:
  - Gender, sex, or sexual orientation
  - Religion, caste, or race
  - Marital status or family dependents
  - Physical disability or health status
  - Geographic pincode proxies designed to infer socioeconomic or community profiles
- Features are strictly limited to behavioral repayment variables (`days_late`, `bounced_emi_count`, `days_past_due`) and objective contract underwriting terms (`loan_amount`, `emi_amount`, `monthly_income`).

### Principle 3: Dignity & Anti-Harassment Safeguards (RBI Fair Practices Code)
All follow-up actions and communications recommended by LoanWatch must comply with the Reserve Bank of India (RBI) Fair Practices Code for Lenders:
- **Calling Hours:** Communication is permitted strictly between **08:00 and 19:00**. Calling during late night or early morning hours is strictly prohibited.
- **Respectful & Supportive Tone:** Suggested action prompts must emphasize assistance and restructuring (e.g., *"Initiate a courteous telephonic check-in to confirm payment arrangement"* or *"Consider a respectful in-person follow-up and restructuring check"*). Threatening, abusive, or coercive language is strictly forbidden.
- **Privacy of Contact:** Calls and field visits must be conducted directly with the borrower or designated co-applicant. Collection officers are strictly barred from contacting employers, neighbors, or acquaintances to shame or pressure borrowers.

### Principle 4: Transparent Explanations ("Warning Signs", Not "Causes")
- To prevent confirmation bias among collectors, the system generates plain-English indicators labeled explicitly as **"warning signs"**, never as absolute "causes" of future failure.
- A borrower flagged for "2 auto-debit bounces" is not condemned as a defaulter; rather, the officer is informed that recent auto-debit presentation failures correlate with cashflow friction, prompting a proactive offer of repayment restructuring.

---

## 3. Synthetic Data & Academic MVP Caveat

> [!WARNING]
> **Synthetic Data Disclaimer:**
> LoanWatch was built and benchmarked using synthetically generated loan trajectories (seeded for reproducibility). No actual customer personal data or proprietary banking records were used.
>
> - **Performance Bounds:** Metrics reported in this project (e.g., PR-AUC of 0.6913, Brier score of 0.0522) reflect behavior on synthetic data distributions and **do NOT represent real-world production performance**.
> - **Production Deployment Prerequisite:** Prior to deploying in an active financial institution, this pipeline must undergo rigorous retraining and fairness audits on historical core banking system (CBS) and loan management system (LMS) records under the supervision of the institutional Credit Risk Committee.

---

## 4. Operational Escalation Protocol

If a borrower disputes an early-warning classification or reports distress due to exceptional circumstances (e.g., documented medical emergency, loss of employment, or macro disaster):
1. **Collector Override:** The collection officer may record interaction notes in the worklist and mark the case as *"Escalated"*.
2. **Restructuring Review:** Escalated accounts are routed to the Credit Restructuring Desk for EMI holiday consideration or tenure extension, pausing active follow-up outreach.
3. **Audit Trail:** All status changes, manual overrides, and timestamps are persisted in the SQLite audit log (`followups` table) for internal risk compliance review.

---

## 5. Authentication & Access Governance Limitations

LoanWatch implements role-based access control and session authentication tailored for an applied synthetic-data MVP:
- **Scope & Design:** Uses standard PBKDF2-HMAC-SHA256 password hashing with salt, rate-limiting lockout (5 failed attempts block for 30 seconds), and server-side collector data isolation.
- **MVP Limitations:** This demo-grade authentication architecture does not incorporate enterprise-grade controls such as Multi-Factor Authentication (MFA), hardware tokens, password complexity rotation policies, self-service recovery, or persistent SIEM security event log streaming.
- **Enterprise Requirement:** Production deployment within regulated financial institutions mandates integrating an enterprise Identity and Access Management (IAM) infrastructure (e.g., Okta, Microsoft Entra ID, SAML/OIDC) with encrypted token rotation and comprehensive access auditing.
