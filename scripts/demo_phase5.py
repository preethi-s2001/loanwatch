import sys
import pandas as pd
from src.predict import score_active_loans
from src.config import SAMPLE_DATA_PATH

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

df = pd.read_csv(SAMPLE_DATA_PATH)
scored = score_active_loans(df)

print(f"Total active loans scored: {len(scored):,}")
print("\nDistribution by Risk Tier:")
print(scored['risk_bucket'].value_counts().to_string())

print("\n=== SAMPLE HIGH RISK LOAN PREDICTION ===")
high_ex = scored[scored['risk_bucket'] == 'High Risk'].iloc[0]
print(f"Loan ID:           {high_ex['loan_id']}")
print(f"Probability:       {high_ex['probability']:.2%}")
print(f"Risk Bucket:       {high_ex['risk_bucket']}")
print(f"Outstanding:       ₹{high_ex['outstanding_amount']:,.2f}")
print(f"DPD:               {high_ex['days_past_due']} days")
print(f"Suggested Action:  {high_ex['suggested_action']}")
print("Warning Signs:")
for s in high_ex['warning_signs']:
    print(f"  • {s}")

print("\n=== SAMPLE MEDIUM RISK LOAN PREDICTION ===")
med_ex = scored[scored['risk_bucket'] == 'Medium Risk'].iloc[0]
print(f"Loan ID:           {med_ex['loan_id']}")
print(f"Probability:       {med_ex['probability']:.2%}")
print(f"Risk Bucket:       {med_ex['risk_bucket']}")
print(f"Outstanding:       ₹{med_ex['outstanding_amount']:,.2f}")
print(f"DPD:               {med_ex['days_past_due']} days")
print(f"Suggested Action:  {med_ex['suggested_action']}")
print("Warning Signs:")
for s in med_ex['warning_signs']:
    print(f"  • {s}")

print("\n=== SAMPLE LOW RISK LOAN PREDICTION ===")
low_ex = scored[scored['risk_bucket'] == 'Low Risk'].iloc[0]
print(f"Loan ID:           {low_ex['loan_id']}")
print(f"Probability:       {low_ex['probability']:.2%}")
print(f"Risk Bucket:       {low_ex['risk_bucket']}")
print(f"Outstanding:       ₹{low_ex['outstanding_amount']:,.2f}")
print(f"DPD:               {low_ex['days_past_due']} days")
print(f"Suggested Action:  {low_ex['suggested_action']}")
print("Warning Signs:")
for s in low_ex['warning_signs']:
    print(f"  • {s}")
