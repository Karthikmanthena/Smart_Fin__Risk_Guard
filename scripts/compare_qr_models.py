import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

# LOAD DATA

df = pd.read_csv("../data/synthetic_qr_transaction_data.csv")

X = df[['amount', 'hour', 'merchant_new', 'txn_count_today']]
y_true = df['anomaly_flag']

results = []

# ISOLATION FOREST

iso = IsolationForest(contamination=0.05, random_state=42)
iso_pred = iso.fit_predict(X)
iso_pred = np.where(iso_pred == -1, 1, 0)

results.append([
    "Isolation Forest",
    accuracy_score(y_true, iso_pred),
    precision_score(y_true, iso_pred),
    recall_score(y_true, iso_pred),
    f1_score(y_true, iso_pred)
])

# ONE-CLASS SVM

svm = OneClassSVM(nu=0.05, gamma='scale')
svm_pred = svm.fit_predict(X)
svm_pred = np.where(svm_pred == -1, 1, 0)

results.append([
    "One-Class SVM",
    accuracy_score(y_true, svm_pred),
    precision_score(y_true, svm_pred),
    recall_score(y_true, svm_pred),
    f1_score(y_true, svm_pred)
])


# LOCAL OUTLIER FACTOR

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_pred = lof.fit_predict(X)
lof_pred = np.where(lof_pred == -1, 1, 0)

results.append([
    "Local Outlier Factor",
    accuracy_score(y_true, lof_pred),
    precision_score(y_true, lof_pred),
    recall_score(y_true, lof_pred),
    f1_score(y_true, lof_pred)
])


# K-MEANS

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_pred = kmeans.fit_predict(X)

# assume smaller cluster = anomaly
cluster_counts = pd.Series(kmeans_pred).value_counts()
anomaly_cluster = cluster_counts.idxmin()
kmeans_pred = np.where(kmeans_pred == anomaly_cluster, 1, 0)

results.append([
    "K-Means",
    accuracy_score(y_true, kmeans_pred),
    precision_score(y_true, kmeans_pred),
    recall_score(y_true, kmeans_pred),
    f1_score(y_true, kmeans_pred)
])

# DISPLAY RESULTS

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print(results_df)
