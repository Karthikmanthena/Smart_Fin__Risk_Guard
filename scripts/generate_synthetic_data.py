

import pandas as pd
import numpy as np

# -----------------------------
# SET RANDOM SEED
# -----------------------------
np.random.seed(42)

# PART 1: SYNTHETIC USER FINANCIAL DATA (RISK PREDICTION)

rows_financial = 2000

monthly_income = np.random.randint(15000, 100000, rows_financial)
emi = (monthly_income * np.random.uniform(0.1, 0.6, rows_financial)).astype(int)
monthly_expense = (monthly_income * np.random.uniform(0.3, 0.9, rows_financial)).astype(int)
savings = monthly_income - (emi + monthly_expense)

risk_level = []

for i in range(rows_financial):
    emi_ratio = emi[i] / monthly_income[i]
    expense_ratio = monthly_expense[i] / monthly_income[i]

    if emi_ratio <= 0.3 and expense_ratio <= 0.6:
        risk_level.append("Low")
    elif emi_ratio <= 0.5 or expense_ratio <= 0.8:
        risk_level.append("Medium")
    else:
        risk_level.append("High")

financial_df = pd.DataFrame({
    "monthly_income": monthly_income,
    "emi": emi,
    "monthly_expense": monthly_expense,
    "savings": savings,
    "risk_level": risk_level
})

# SAVE FILE
financial_df.to_csv("../data/synthetic_user_financial_data.csv", index=False)

print("✔ Synthetic User Financial Data Created")


# PART 2: SYNTHETIC QR TRANSACTION DATA (FRAUD DETECTION)

rows_qr = 3000

amount = np.random.randint(50, 50000, rows_qr)
hour = np.random.randint(0, 24, rows_qr)
merchant_new = np.random.choice([0, 1], rows_qr, p=[0.7, 0.3])
txn_count_today = np.random.randint(1, 8, rows_qr)

anomaly_flag = []

for i in range(rows_qr):
    if (
        amount[i] > 30000 or
        (merchant_new[i] == 1 and hour[i] < 6) or
        txn_count_today[i] > 5
    ):
        anomaly_flag.append(1)
    else:
        anomaly_flag.append(0)

qr_df = pd.DataFrame({
    "amount": amount,
    "hour": hour,
    "merchant_new": merchant_new,
    "txn_count_today": txn_count_today,
    "anomaly_flag": anomaly_flag
})

# SAVE FILE
qr_df.to_csv("../data/synthetic_qr_transaction_data.csv", index=False)

print("✔ Synthetic QR Transaction Data Created")
