import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# LOAD DATA
data = pd.read_csv("../data/synthetic_user_financial_data.csv")


# FEATURES & TARGET

X = data[["monthly_income", "emi", "monthly_expense", "savings"]]
y = data["risk_level"]


# TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RANDOM FOREST MODEL

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# PREDICTION & EVALUATION

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL
joblib.dump(rf_model, "../models/random_forest_risk_model.pkl")

print("\n✔ Random Forest Model Trained and Saved")
