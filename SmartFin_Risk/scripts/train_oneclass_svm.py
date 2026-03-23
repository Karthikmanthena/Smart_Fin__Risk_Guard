import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
import os

qr_df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

X = qr_df[['amount', 'hour', 'merchant_new', 'txn_count_today']]

model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
model.fit(X)

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/oneclass_svm_qr.pkl")

print("✔ One-Class SVM trained & saved")