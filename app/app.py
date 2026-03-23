import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import joblib
import os
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SmartFin RiskGuard",
    page_icon="💳",
    layout="wide"
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

risk_model = joblib.load(os.path.join(BASE_DIR, "../models/financial_risk_model.pkl"))
fraud_model = joblib.load(os.path.join(BASE_DIR, "../models/fraud_detection_model.pkl"))
# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
username TEXT PRIMARY KEY,
password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS user_finance(
username TEXT PRIMARY KEY,
income REAL,
emi REAL,
expense REAL,
savings REAL
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS expenses(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
category TEXT,
amount REAL,
date TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS payment_history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,
merchant TEXT,
amount REAL,
date TEXT,
risk_score REAL
)
""")

conn.commit()

# ---------------- USER FUNCTIONS ----------------
def register_user(username,password):
    try:
        c.execute("INSERT INTO users VALUES (?,?)",(username,password))
        conn.commit()
        return True
    except:
        return False

def login_user(username,password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(username,password))
    return c.fetchone()

def reset_password(username,new_password):
    c.execute("UPDATE users SET password=? WHERE username=?",(new_password,username))
    conn.commit()

# ---------------- FINANCE ----------------
def save_finance(username,income,emi,expense,savings):
    c.execute("INSERT OR REPLACE INTO user_finance VALUES (?,?,?,?,?)",
              (username,income,emi,expense,savings))
    conn.commit()

def load_finance(username):
    c.execute("SELECT * FROM user_finance WHERE username=?", (username,))
    return c.fetchone()

# ---------------- EXPENSE ----------------
def save_expense(username,category,amount):
    c.execute("INSERT INTO expenses VALUES (NULL,?,?,?,?)",
              (username,category,amount,str(datetime.now())))
    conn.commit()

def load_expenses(username):
    return pd.read_sql_query(
        "SELECT category,amount,date FROM expenses WHERE username=?",
        conn, params=(username,)
    )

# ---------------- PAYMENTS ----------------
def load_payments(username):
    return pd.read_sql_query(
        "SELECT merchant,amount,date,risk_score FROM payment_history WHERE username=?",
        conn, params=(username,)
    )

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
def login_page():

    st.title("💳 SmartFin RiskGuard")

    menu = ["Login","Register","Reset Password"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Login":

        username = st.text_input("Username")
        password = st.text_input("Password",type="password")

        if st.button("Login"):

            if login_user(username,password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    elif choice == "Register":

        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type="password")

        if st.button("Register"):

            if register_user(new_user,new_password):
                st.success("Account Created")
            else:
                st.warning("Username exists")

    elif choice == "Reset Password":

        username = st.text_input("Username")
        new_pass = st.text_input("New Password",type="password")

        if st.button("Reset"):
            reset_password(username,new_pass)
            st.success("Password Updated")

# ---------------- ADD EXPENSE ----------------
def add_expense():

    st.header("💰 Add Expense")

    category = st.selectbox(
        "Category",
        ["Food","Shopping","Bills","Transport","Entertainment","Other"]
    )

    amount = st.number_input("Amount",min_value=0)

    if st.button("Add Expense"):
        save_expense(st.session_state.user,category,amount)
        st.success("Saved")

# ---------------- DASHBOARD ----------------
def dashboard():

    st.title("📊 Dashboard")

    df = load_expenses(st.session_state.user)

    if df.empty:
        st.info("No data")
        return

    df["date"] = pd.to_datetime(df["date"])
    today = pd.Timestamp.now()

    week = df[df["date"] >= today - pd.Timedelta(days=7)]
    month = df[df["date"] >= today - pd.Timedelta(days=30)]

    st.metric("Weekly Expense", int(week["amount"].sum()))
    st.metric("Monthly Expense", int(month["amount"].sum()))

    st.plotly_chart(px.pie(week, names="category", values="amount"))
    st.plotly_chart(px.bar(month, x="category", y="amount"))

# ---------------- FINANCIAL RISK ----------------
def financial_risk():

    st.header("📈 Risk Prediction")

    income = st.number_input("Income")
    emi = st.number_input("EMI")
    expense = st.number_input("Expense")
    savings = st.number_input("Savings")

    if st.button("Predict"):

        df = pd.DataFrame([[income,emi,expense,savings]],
        columns=["monthly_income","emi","monthly_expense","savings"])

        result = risk_model.predict(df)[0]

        st.success(f"Risk: {result}")

# ---------------- QR SCANNER ----------------
def qr_scanner():

    st.header("🔍 QR Fraud Detection")

    file = st.file_uploader("Upload QR",type=["png","jpg","jpeg"])

    if file:

        img = cv2.imdecode(np.frombuffer(file.read(),np.uint8),1)
        st.image(img)

        detector = cv2.QRCodeDetector()
        data,_,_ = detector.detectAndDecode(img)

        if data:

            st.code(data)

            amount = st.number_input("Amount")

            sample = pd.DataFrame(
                [[amount,12,1,5]],
                columns=["amount","hour","merchant_new","txn_count_today"]
            )

            result = fraud_model.predict(sample)[0]

            if result == -1:
                st.error("Fraud Detected")
            else:
                st.success("Safe")

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login_page()
else:

    menu = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard","Add Expense","Financial Risk","QR Scanner","Logout"]
    )

    if menu == "Dashboard":
        dashboard()

    elif menu == "Add Expense":
        add_expense()

    elif menu == "Financial Risk":
        financial_risk()

    elif menu == "QR Scanner":
        qr_scanner()

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()