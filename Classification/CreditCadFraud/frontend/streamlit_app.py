import streamlit as st
import requests

# Backend FastAPI URL
FASTAPI_URL = "http://127.0.0.1:8000"   # Update if deployed elsewhere

# ----------------- Streamlit UI ----------------
st.set_page_config(page_title="Fraud Detection", page_icon="🔒", layout="centered")

st.title("🔒 Fraud Detection System")

# Session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------- Login Page ----------------
if not st.session_state.logged_in:
    st.subheader("Login to Continue")
    uname = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        response = requests.post(f"{FASTAPI_URL}/login", json={"uname": uname, "password": password})
        if response.status_code == 200 and response.json() == "login successfully ":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
        else:
            st.error("❌ Incorrect username or password")

# ----------------- Input Page ----------------
if st.session_state.logged_in:
    st.subheader("Enter Transaction Details")

    Failed_Transaction_Count_7d = st.number_input("Failed Transaction Count (7d)", min_value=0, step=1)
    Risk_Score = st.number_input("Risk Score", min_value=0, step=1)
    IP_Address_Flag = st.selectbox("IP Address Flag", [0, 1])
    Transaction_Amount = st.number_input("Transaction Amount", min_value=0, step=1)
    Previous_Fraudulent_Activity = st.selectbox("Previous Fraudulent Activity", [0, 1])
    Timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value="2025-08-16 12:00:00")

    if st.button("Predict"):
        payload = {
            "Failed_Transaction_Count_7d": Failed_Transaction_Count_7d,
            "Risk_Score": Risk_Score,
            "IP_Address_Flag": IP_Address_Flag,
            "Transaction_Amount": Transaction_Amount,
            "Previous_Fraudulent_Activity": Previous_Fraudulent_Activity,
            "Timestamp": Timestamp
        }

        response = requests.post(f"{FASTAPI_URL}/input", json=payload)
        if response.status_code == 200:
            result = response.json().get("result", "Error")
            st.success(f"🔮 Prediction Result: **{result}**")
        else:
            st.error("⚠️ Error during prediction")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.info("Logged out successfully")
