import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import os
from dotenv import load_dotenv
load_dotenv()

# ---- Initialize Firebase ----
# cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# cred = credentials.Certificate(cred_path)
# if not firebase_admin._apps:
#     firebase_admin.initialize_app(cred)

firebase_credentials = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"],
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    "universe_domain": st.secrets["firebase"]["universe_domain"],
}

cred = credentials.Certificate(firebase_credentials)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="Sign Up | BridgeSign", page_icon="🧏‍♀️", layout="wide")

# ---- Page Background and Custom Styles ----
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f5ed !important;
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

col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col1:
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)

with col2:
    # ---- Page Content ----
    st.markdown(
        "<h1 style='color:#0077B6; text-align: center;'>Create Your HandsIn Account!</h1>",
        unsafe_allow_html=True
    )

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

with col3:
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)


# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
