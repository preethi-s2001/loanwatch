# LoanWatch Risk Threshold Rationale & Decision Framework

## 1. Executive Summary
In retail and MSME Non-Banking Financial Companies (NBFCs), risk score cut-offs cannot be chosen arbitrarily based on academic defaults (such as 0.50). Cut-offs directly dictate:
1. **Collector Workload & Capacity Utilization:** How many accounts appear on daily worklists.
2. **Contact Fatigue & Customer Goodwill:** The risk of harassing borrowers who would have cured on their own.
3. **Loss Mitigation vs Operational Cost:** Balancing the cost of missed defaults against the variable cost of collection outreach.

This document outlines the rationale behind the default LoanWatch thresholds and provides a structured **decision template** for tuning thresholds with risk committees and operations heads.

---

## 2. Default Configuration & Operational Tiers

LoanWatch categorizes active accounts ($< 90$ DPD) into three operational tiers:

```
[0.00 ----------------------- 0.30 ----------------------- 0.60 ----------------------- 1.00]
        LOW RISK                        MEDIUM RISK                     HIGH RISK
 (Digital/Automated Reminders)     (Telephonic Outreach)           (Field Visit & Restructuring)
```

| Risk Tier | Probability Range ($\hat{p}$) | Primary Operational Channel | Respectful Action Guidance | Expected Workload Share |
| :--- | :---: | :---: | :--- | :---: |
| **High Risk** | **$\ge 0.60$** | In-Person Senior Field Visit | *"Consider a respectful in-person follow-up and restructuring check"* | ~5% – 8% of portfolio |
| **Medium Risk** | **$0.30 \le \hat{p} < 0.60$** | Telephonic Contact Center | *"Initiate a courteous telephonic check-in to confirm payment arrangement"* | ~12% – 18% of portfolio |
| **Low Risk** | **$< 0.30$** | Automated Digital Push | *"Send standard automated SMS/App payment reminder"* | ~75% – 82% of portfolio |

---

## 3. Cost-Benefit Trade-Off Matrix

### The Economics of Early Warning Outreach

$$\text{Net Value}(\tau) = \sum_{i \in \text{Worklist}(\tau)} \left( \hat{p}_i \times \text{EAD}_i \times \text{LGD} \times \text{Cure Rate Lift} \right) - \sum_{i \in \text{Worklist}(\tau)} \text{Cost of Contact}$$

Where:
- $\text{EAD}_i$: Exposure at Default (Outstanding balance).
- $\text{LGD}$: Loss Given Default (typically 50% to 70% in unsecured retail/MSME lending).
- $\text{Cure Rate Lift}$: The incremental fraction of defaulting accounts saved by early human outreach (~25% to 40%).
- $\text{Cost of Contact}$:
  - Field visit: ₹250 – ₹500 per in-person visit (fuel, collector compensation).
  - Telephonic call: ₹15 – ₹35 per dial / conversation.
  - Digital SMS/WhatsApp/Push: ₹0.15 – ₹0.50 per automated alert.

### Error Consequences:
- **False Negative (Missed Defaulter):** Account deteriorates from 30 DPD to 90+ DPD (NPA) without early engagement. Recovery becomes legally intensive, and expected loss is high (~₹40,000 to ₹1,50,000 on average ticket size).
- **False Positive (Unnecessary Outreach):** Collector visits or calls a borrower who was merely temporarily delayed and would have cleared the balance independently. Consequences: wasted collector time (~₹350) and borrower dissatisfaction.

> [!NOTE]
> In retail credit, the cost ratio of a **False Negative to a False Positive is approximately 150:1 to 300:1**. Therefore, optimal early-warning cut-offs skew towards higher sensitivity (lower thresholds like 0.30–0.35) than a balanced 0.50.

---

## 4. Operational Capacity Constraints

Regardless of statistical optimality, physical field capacity imposes a hard constraint on high-risk thresholds:

$$\text{Accounts Flagged as High Risk} \le N_{\text{Field Collectors}} \times \text{Visits per Collector per Month}$$

- **Industry Benchmark:** A full-time collection field executive handles **8 to 12 in-person borrower visits per day** (~200 to 250 visits/month).
- **Capacity Formula for Risk Committees:**
  $$\text{Max Allowed High-Risk Volume} = \text{Active Field Force} \times 220$$
  If the High Risk cut-off of 0.60 generates 800 accounts, but the team has only 2 field officers (max 440 visits), the High Risk threshold **must be raised** (e.g., to 0.70) to prevent backlog and collector burnout, while accounts between 0.50–0.70 are routed to tele-calling.

---

## 5. Committee Decision Template (For Risk Head & Industry Mentor)

When presenting to credit committees or calibrating with mentors, use this 4-step framework:

1. **Step 1: Document Capacity**
   - Active field collectors: `_____`
   - Active tele-callers: `_____`
   - Monthly field capacity ($N \times 220$): `_____` accounts
2. **Step 2: Review Holdout Test Split Confusion Matrix at Candidate Thresholds**
   - At $\tau = 0.25$: High Recall (85%), but higher False Positives.
   - At $\tau = 0.35$: Balanced sweet spot (Recall 78%, Precision 57%, F1 0.66).
   - At $\tau = 0.50$: Conservative (Recall 58%, Precision 72%), misses 42% of future defaults.
3. **Step 3: Agree on Escalation Policy**
   - Agree that automated alerts trigger for all accounts with $\hat{p} \ge 0.15$.
   - Agree that telephonic outreach handles $0.30 \le \hat{p} < 0.60$.
   - Agree that field visits are reserved strictly for $\hat{p} \ge 0.60$ with outstanding amount $> ₹50,000$.
4. **Step 4: Audit & Responsible-Use Sign-Off**
   - Reaffirm that no account is subjected to automated punitive measures.
   - Reaffirm that outreach follows RBI Fair Practices Code for Lenders (calling hours between 08:00 and 19:00, dignified communication, no harassment).
