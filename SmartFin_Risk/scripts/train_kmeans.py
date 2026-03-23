import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os

qr_df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

X = qr_df[['amount', 'hour', 'merchant_new', 'txn_count_today']]

model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/kmeans_qr.pkl")

print("✔ K-Means model trained & saved")