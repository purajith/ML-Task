# title/frontend/streamlit_app.py

import streamlit as st   # Streamlit for frontend UI
import requests          # For sending HTTP requests to FastAPI backend

# Backend base URL (adjust if backend is on a different host or port)
BACKEND_URL = "http://127.0.0.1:8000"

# Set page configuration: title, icon, and layout
st.set_page_config(
    page_title="Email Spam-Ham Classification",  # Browser tab title
    page_icon="📧",                               # Icon shown in tab
    layout="centered"                             # Centered layout
)
# Main title of the app
st.title("📧 Email Spam-Ham Classifier")

# ------------------ LOGIN SECTION ------------------
st.subheader("🔑 Login")  # Section heading

# Create a form for login so user enters ID & password together
with st.form("login_form"):
    userid = st.text_input("User ID")                       # Input field for User ID
    password = st.text_input("Password", type="password")   # Password input (hidden text)
    login_submit = st.form_submit_button("Login")           # Submit button for the form

# If the Login button is clicked
if login_submit:
    # Check if both fields are filled
    if not userid or not password:
        st.error("Please enter both User ID and Password.")  # Show error if missing
    else:
        try:
            # Send POST request to FastAPI backend /login
            resp = requests.post(f"{BACKEND_URL}/login", json={
                "userid": userid,
                "password": password
            })

            # If backend responds OK
            if resp.status_code == 200:
                st.session_state["login_msg"] = resp.json()["message"]  # Store login result in session
                st.success(st.session_state["login_msg"])               # Show success message
            else:
                st.error("Login request failed.")  # If backend sends error status
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")  # Network or server error

# ------------------ PREDICTION SECTION ------------------
# Show prediction form only if login was successful
if st.session_state.get("login_msg") == "Is correct":
    st.subheader("📨 Email Prediction")  # Section heading

    # Create a form for prediction
    with st.form("predict_form"):
        email_text = st.text_area("Enter email text here...")  # Multiline input for email
        predict_submit = st.form_submit_button("Predict")      # Submit button

    # If Predict button is clicked
    if predict_submit:
        # Check if email content is empty
        if not email_text.strip():
            st.warning("Please enter email content.")  # Warn user to fill content
        else:
            try:
                # Send POST request to FastAPI backend /predict
                resp = requests.post(f"{BACKEND_URL}/predict", json={"email": email_text})

                # If backend responds OK
                if resp.status_code == 200:
                    # Show prediction result (backend returns a string in your case)
                    st.success(f"Prediction: {resp.text}")
                else:
                    st.error("Prediction request failed.")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
