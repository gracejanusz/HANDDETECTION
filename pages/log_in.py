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

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
cred = credentials.Certificate(cred_path)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="Log In | BridgeSign", page_icon="🧏‍♀️", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
    background-color: white !important;
    }
    .stButton>button {
        color: black !important;
        background: #ffe9a5 !important; /* Blue */
        border-radius: 8px !important;
        height: 3em !important;
        width: 100% !important;
        font-size: 1.2em !important;
        margin-top: 10px !important;
    }
    .stButton>button:hover {
        background: #ffd96b !important; /* Orange */
        color: black !important;
    }
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #ffffff !important; /* Pure white input boxes */
        color: black !important; /* Typing is black */
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1em !important;
    }
    /* Label styling */
    label {
        color: #0077B6 !important; /* Make the field labels (Email, Password) blue */
        font-weight: bold !important;
    }
    </style>
          """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='color:#0077B6; text-align: center;'>Welcome Back!</h1>",
    unsafe_allow_html=True
)

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