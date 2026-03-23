import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/synthetic_user_financial_data.csv")

# Features
X = df[['monthly_income','emi','monthly_expense','savings']]
y = df['risk_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "../models/financial_risk_model.pkl")

print("Financial Risk Model Saved")