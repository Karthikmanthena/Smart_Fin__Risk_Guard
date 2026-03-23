import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import joblib
import os

qr_df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

X = qr_df[['amount', 'hour', 'merchant_new', 'txn_count_today']]

model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

outlier_scores = model.fit_predict(X)

os.makedirs("../models", exist_ok=True)
joblib.dump(outlier_scores, "../models/lof_qr.pkl")

print("✔ LOF model evaluated & scores saved")