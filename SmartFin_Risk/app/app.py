import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import joblib
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SmartFin RiskGuard",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
risk_model = joblib.load("../models/financial_risk_model.pkl")
fraud_model = joblib.load("../models/fraud_detection_model.pkl")

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

# ---------------- FINANCE FUNCTIONS ----------------
def save_finance(username,income,emi,expense,savings):
    c.execute("""
    INSERT OR REPLACE INTO user_finance VALUES (?,?,?,?,?)
    """,(username,income,emi,expense,savings))
    conn.commit()

def load_finance(username):
    c.execute("SELECT * FROM user_finance WHERE username=?", (username,))
    return c.fetchone()

# ---------------- EXPENSE FUNCTIONS ----------------
def save_expense(username,category,amount):
    c.execute("""
    INSERT INTO expenses (username,category,amount,date)
    VALUES (?,?,?,?)
    """,(username,category,amount,str(datetime.now())))
    conn.commit()

def load_expenses(username):
    query = """
    SELECT category,amount,date
    FROM expenses
    WHERE username=?
    """
    return pd.read_sql_query(query,conn,params=(username,))

# ---------------- PAYMENT HISTORY ----------------
def load_payments(username):

    query = """
    SELECT merchant,amount,date,risk_score
    FROM payment_history
    WHERE username=?
    """

    return pd.read_sql_query(query,conn,params=(username,))

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

            user = login_user(username,password)

            if user:
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
                st.warning("Username already exists")

    elif choice == "Reset Password":

        username = st.text_input("Username")
        new_pass = st.text_input("New Password",type="password")

        if st.button("Reset"):
            reset_password(username,new_pass)
            st.success("Password Updated")

# ---------------- ADD EXPENSE ----------------
def add_expense():

    st.header("💰 Add Daily Expense")

    category = st.selectbox(
        "Category",
        ["Food","Shopping","Bills","Transport","Entertainment","Other"]
    )

    amount = st.number_input("Amount",min_value=0)

    if st.button("Add Expense"):

        save_expense(
            st.session_state.user,
            category,
            amount
        )

        st.success("Expense recorded")

# ---------------- DASHBOARD ----------------
def dashboard():

    st.title("📊 Expense Dashboard")

    df = load_expenses(st.session_state.user)

    if df.empty:
        st.info("No expenses yet")
        return

    df["date"] = pd.to_datetime(df["date"])
    today = pd.Timestamp.now()

    week_data = df[df["date"] >= today - pd.Timedelta(days=7)]
    month_data = df[df["date"] >= today - pd.Timedelta(days=30)]

    weekly_total = week_data["amount"].sum()
    monthly_total = month_data["amount"].sum()

    col1,col2 = st.columns(2)
    col1.metric("Weekly Expenses",int(weekly_total))
    col2.metric("Monthly Expenses",int(monthly_total))

    st.subheader("Weekly Expense Distribution")

    fig1 = px.pie(
        week_data,
        names="category",
        values="amount"
    )

    st.plotly_chart(fig1,use_container_width=True)

    st.subheader("Monthly Expense Distribution")

    fig2 = px.bar(
        month_data,
        x="category",
        y="amount",
        color="category"
    )

    st.plotly_chart(fig2,use_container_width=True)

# ---------------- FINANCIAL RISK ----------------
def financial_risk():

    st.header("📈 Financial Risk Prediction")

    data = load_finance(st.session_state.user)

    if data:
        income = st.number_input("Income",value=data[1])
        emi = st.number_input("EMI",value=data[2])
        expense = st.number_input("Expense",value=data[3])
        savings = st.number_input("Savings",value=data[4])
    else:
        income = st.number_input("Income")
        emi = st.number_input("EMI")
        expense = st.number_input("Expense")
        savings = st.number_input("Savings")

    if st.button("Save Financial Details"):
        save_finance(st.session_state.user,income,emi,expense,savings)
        st.success("Saved")

    if st.button("Predict Risk"):

        df = pd.DataFrame([[income,emi,expense,savings]],
        columns=["monthly_income","emi","monthly_expense","savings"])

        risk = risk_model.predict(df)[0]

        st.subheader("Risk Result")

        if risk == "Low":
            st.success("Low Financial Risk")

        elif risk == "Medium":
            st.warning("Medium Financial Risk")

        else:
            st.error("High Financial Risk")

        # Financial Health Score
        if income > 0:
            score = (savings / income) * 100
            st.subheader("Financial Health Score")
            st.progress(min(int(score),100))
            st.write(f"{round(score,1)} / 100")

# ---------------- QR FRAUD DETECTION ----------------
def qr_scanner():

    st.header("🔍 QR Fraud Detection")

    uploaded_file = st.file_uploader("Upload QR Image",type=["png","jpg","jpeg"])

    if uploaded_file:

        file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        img = cv2.imdecode(file_bytes,1)

        st.image(img,use_container_width=True)

        detector = cv2.QRCodeDetector()
        data,bbox,_ = detector.detectAndDecode(img)

        if data:

            st.code(data)

            merchant = "Unknown"

            if "pn=" in data:
                merchant = data.split("pn=")[1].split("&")[0]

            st.write("Merchant:",merchant)

            amount = st.number_input("Payment Amount")

            hour = pd.Timestamp.now().hour

            merchant_new = st.selectbox("New Merchant",[0,1])

            txn_today = st.number_input("Transactions Today",0,20)

            if st.button("Analyze QR"):

                sample = pd.DataFrame(
                [[amount,hour,merchant_new,txn_today]],
                columns=["amount","hour","merchant_new","txn_count_today"]
                )

                result = fraud_model.predict(sample)[0]
                risk_score = fraud_model.decision_function(sample)[0]

                st.metric("Fraud Risk Score",round(risk_score,3))

                if result == -1:
                    st.error("🚨 Suspicious QR")

                else:
                    st.success("QR appears safe")

                c.execute("""
                INSERT INTO payment_history(username,merchant,amount,date,risk_score)
                VALUES (?,?,?,?,?)
                """,(st.session_state.user,merchant,amount,str(datetime.now()),risk_score))

                conn.commit()

# ---------------- PAYMENT HISTORY ----------------
def payment_dashboard():

    st.header("💳 Payment History")

    df = load_payments(st.session_state.user)

    if df.empty:
        st.info("No payments yet")
        return

    df["date"] = pd.to_datetime(df["date"])

    st.dataframe(df)

    st.subheader("Merchant Spending")

    fig = px.bar(
        df,
        x="merchant",
        y="amount",
        color="merchant"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Fraud Risk Trend")

    fig2 = px.line(
        df,
        x="date",
        y="risk_score"
    )

    st.plotly_chart(fig2,use_container_width=True)

# ---------------- MAIN APP ----------------
if not st.session_state.logged_in:

    login_page()

else:

    st.sidebar.success(f"Logged in as {st.session_state.user}")

    menu = st.sidebar.selectbox(
        "Navigation",
        [
        "Dashboard",
        "Add Expense",
        "Financial Risk",
        "QR Fraud Detection",
        "Payment History",
        "Logout"
        ]
    )

    if menu == "Dashboard":
        dashboard()

    elif menu == "Add Expense":
        add_expense()

    elif menu == "Financial Risk":
        financial_risk()

    elif menu == "QR Fraud Detection":
        qr_scanner()

    elif menu == "Payment History":
        payment_dashboard()

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()