import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

qr_df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

X = qr_df[['amount', 'hour', 'merchant_new', 'txn_count_today']]

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/isolation_forest_qr.pkl")

print("✔ Isolation Forest trained & saved")