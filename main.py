import streamlit as st
from PIL import Image
import firebase_admin
from firebase_admin import credentials

# ---- Initialize Firebase (only once) ----
if not firebase_admin._apps:
    cred = credentials.Certificate('handsinactual-5abb2c6c4c1c.json')
    firebase_admin.initialize_app(cred)

# ---- Set page config ----
st.set_page_config(page_title="BridgeSign", page_icon="🧏‍♀️", layout="centered")

# ---- CUSTOM CSS for background and button styling ----
st.markdown(
    """
    <style>
    body {
        background-color: #f7f5ed;
    }
    .stApp {
        background-color: #f7f5ed;
    }
    div.stButton > button {
        background-color: #ffe9a5;
        color: black;
        border: none;
        padding: 0.75em 2em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        margin-bottom: 0.5em;
    }
    div.stButton > button:hover {
        background-color: #ffd96b;
        color: black;
    }
    .mission-button > button {
        background-color: #0277b5;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        width: 40%;
        min-width: 150px;
        max-width: 200px;
        margin-top: 1em;
        cursor: pointer;
    }
    .mission-button > button:hover {
        background-color: #026099;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Big Logo at the Top ----
st.image("pictures/big_logo.png", use_container_width=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("BridgeSign")
    st.markdown("**Empowering Communication**")

# ---- Main Title and Description ----
st.markdown(
    """
    <h1 style='text-align: center; color: #0277b5;'>The AI Sign Language Trainer for Healthcare</h1>
    <h4 style='text-align: center; color: #2aaaff;'>An interactive platform helping professionals build essential ASL skills through AI-driven practice.</h4>
    """,
    unsafe_allow_html=True
)

# ---- Smaller, Centered Our Mission Button ----
st.markdown("<div class='mission-button' style='text-align: center;'>", unsafe_allow_html=True)

if st.button("Our Mission", key="mission_button"):
    st.switch_page("pages/mission.py")

st.markdown("</div>", unsafe_allow_html=True)

# ---- Add vertical space before Login and Sign Up buttons ----
st.markdown("<br><br>", unsafe_allow_html=True)

# ---- Centered Call to Action: LOGIN and SIGN UP Buttons ----
st.markdown("<div style='text-align: center; max-width: 400px; margin: auto;'>", unsafe_allow_html=True)

if st.button("Login", key="login_button"):
    st.switch_page("pages/log_in.py")

if st.button("Sign Up", key="signup_button"):
    st.switch_page("pages/sign_up.py")

st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
