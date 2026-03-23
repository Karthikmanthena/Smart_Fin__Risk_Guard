import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

data = pd.read_csv("../data/synthetic_user_financial_data.csv")

X = data[["monthly_income", "emi", "monthly_expense", "savings"]]
y = data["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, pred))

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/decision_tree_model.pkl")