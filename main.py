import streamlit as st
from PIL import Image
import firebase_admin
from firebase_admin import credentials

# ---- Initialize Firebase (only once) ----
if not firebase_admin._apps:
    cred = credentials.Certificate('handsin-e15d2-7fe9c1f743a4.json')
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

# ---- Main Title ----
st.markdown(
    """
    <h1 style='text-align: center; color: #0277b5;'>The AI Sign Language Trainer for Healthcare</h1>
    <h4 style='text-align: center; color: #2aaaff;'>An interactive platform helping professionals build essential ASL skills through AI-driven practice.</h4>
    """,
    unsafe_allow_html=True
)

# ---- Add vertical space before buttons ----
st.markdown("<br><br>", unsafe_allow_html=True)

# ---- Centered Call to Action: SIGN UP and LOGIN Buttons ----
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.empty()

with col2:
    # Sign Up Button
    if st.button("Sign Up", key="signup_button"):
        st.switch_page("pages/sign_up.py")  # ✅ Navigates to Sign Up page


with col3:
    st.empty()

# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
