# pages/log_in.py

import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import requests
from dotenv import load_dotenv
import os

# ---- Load Environment Variables ----
load_dotenv()

# ---- Initialize Firebase ----
cred = credentials.Certificate('handsinactual-5abb2c6c4c1c.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="Log In | BridgeSign", page_icon="🧏‍♀️", layout="centered")

st.title("Welcome Back!")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

login_button = st.button("Log In")

if login_button:
    try:
        # Securely load API key
        firebase_api_key = os.getenv("FIREBASE_API_KEY")

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        response = requests.post(url, json=payload)
        result = response.json()

        if "idToken" in result:
            st.success("Login successful! 🎉 Redirecting...")
            st.session_state["user"] = result
            st.switch_page("pages/library.py")
        else:
            st.error(f"Login failed: {result.get('error', {}).get('message', 'Unknown error')}")

    except Exception as e:
        st.error(f"Login failed: {e}")

else:
    if st.button("Back to Home", key="back_home_button_login"):
        st.switch_page("main.py")
