import pandas as pd
from src.config import SAMPLE_DATA_PATH, EMBARGO_MONTHS, TARGET_COLUMN, get_split_dates

df = pd.read_csv(SAMPLE_DATA_PATH)
df['obs_dt'] = pd.to_datetime(df['observation_date'])
unique_dates = sorted(df['obs_dt'].unique())

split_dates = get_split_dates(unique_dates)

train_dates = split_dates["train"]
embargo1_dates = split_dates["embargo_1"]
val_dates = split_dates["validation"]
embargo2_dates = split_dates["embargo_2"]
test_dates = split_dates["test"]

df_train = df[df['obs_dt'].isin(train_dates)]
df_val = df[df['obs_dt'].isin(val_dates)]
df_test = df[df['obs_dt'].isin(test_dates)]

print("=== TIME-BASED SPLIT SUMMARY (WITH 3-MONTH EMBARGO GAPS) ===")
print(f"Total Dates:  {len(unique_dates)} distinct months from {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
print(f"Train Window: {len(train_dates)} months ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')})")
print(f"Embargo 1:    {len(embargo1_dates)} months ({embargo1_dates[0].strftime('%Y-%m-%d')} to {embargo1_dates[-1].strftime('%Y-%m-%d')})")
print(f"Val Window:   {len(val_dates)} months ({val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')})")
print(f"Embargo 2:    {len(embargo2_dates)} months ({embargo2_dates[0].strftime('%Y-%m-%d')} to {embargo2_dates[-1].strftime('%Y-%m-%d')})")
print(f"Test Window:  {len(test_dates)} months ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")

print("\n=== CLASS BALANCE & SAMPLE SIZE PER SPLIT ===")
for name, split_df in [("Train", df_train), ("Validation", df_val), ("Test (OOT)", df_test)]:
    pos = int(split_df[TARGET_COLUMN].sum())
    total = len(split_df)
    loans = split_df["loan_id"].nunique()
    rate = (pos / total * 100) if total > 0 else 0
    print(f"{name:12s} | Rows: {total:6d} | Distinct Loans: {loans:5d} | Positives (Default): {pos:5d} ({rate:5.2f}%)")

pos_test = int(df_test[TARGET_COLUMN].sum())
print(f"\nTest split >= 300 positives check: {pos_test >= 300} (Found: {pos_test})")
print("\n=== DATASET HEAD (5 ROWS) ===")
cols_to_show = [
    "loan_id", "observation_date", "loan_amount", "outstanding_amount",
    "emi_amount", "monthly_income", "days_late", "bounced_emi_count",
    "days_past_due", "collector_id", "default_next_3_months"
]
print(df[cols_to_show].head(5).to_string())
