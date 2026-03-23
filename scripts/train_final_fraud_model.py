import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

# Features
X = df[['amount','hour','merchant_new','txn_count_today']]

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "../models/fraud_detection_model.pkl")

print("Fraud Detection Model Saved")