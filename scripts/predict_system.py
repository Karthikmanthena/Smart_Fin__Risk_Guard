import joblib
import pandas as pd
import numpy as np

# Load trained models
risk_model = joblib.load("../models/financial_risk_model.pkl")
fraud_model = joblib.load("../models/fraud_detection_model.pkl")


# Financial Risk Prediction
def predict_financial_risk(income, emi, expense, savings):

    data = pd.DataFrame([[income, emi, expense, savings]],
                        columns=['monthly_income','emi','monthly_expense','savings'])

    prediction = risk_model.predict(data)

    return prediction[0]


# Fraud Detection
def detect_fraud(amount, hour, merchant_new, txn_count):

    data = pd.DataFrame([[amount, hour, merchant_new, txn_count]],
                        columns=['amount','hour','merchant_new','txn_count_today'])

    result = fraud_model.predict(data)

    if result[0] == -1:
        return "Suspicious Transaction"
    else:
        return "Transaction Normal"