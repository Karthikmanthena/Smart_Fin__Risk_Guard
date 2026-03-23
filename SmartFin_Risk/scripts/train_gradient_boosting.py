import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


data = pd.read_csv("../data/synthetic_user_financial_data.csv")

X = data[["monthly_income", "emi", "monthly_expense", "savings"]]
y = data["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, pred))

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/gradient_boosting_model.pkl")