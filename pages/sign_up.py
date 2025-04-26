import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import os

# ---- Initialize Firebase ----
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
cred = credentials.Certificate(cred_path)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="Sign Up | BridgeSign", page_icon="🧏‍♀️", layout="centered")

st.title("Create Your HandsIn Account!")

email = st.text_input("Email")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

signup_button = st.button("Sign Up")

if signup_button:
    if password != confirm_password:
        st.error("Passwords do not match.")
    elif len(password) < 6:
        st.error("Password must be at least 6 characters long.")
    else:
        try:
            user = auth.create_user(
                email=email,
                password=password
            )
            st.success("Account created successfully! 🎉 You can now log in.")
            st.session_state["signup_successful"] = True
            st.switch_page("pages/log_in.py")

        except Exception as e:
            st.error(f"Signup failed: {e}")

else:
    if st.button("Back to Home", key="back_home_button"):
        st.switch_page("main.py")