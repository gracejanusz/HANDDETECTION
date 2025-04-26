import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="BridgeSign", page_icon="🧏‍♀️", layout="centered")

# ---- CUSTOM CSS for background and button styling ----
# ---- CUSTOM CSS for background and button styling ----
st.markdown(
    """
    <style>
    body {
        background-color: #f7f5ed;
        background-color: #f7f5ed;
    }
    .stApp {
        background-color: #f7f5ed;
    }

    /* Style the Sign Up button */
    div.stButton > button {
        background-color: #ffe9a5;
        color: black;
        border: none;
        padding: 0.75em 2em; /* Bigger button */
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 100%; /* Make button fill the column */
    }
    div.stButton > button:hover {
        background-color: #ffd96b;
        color: black;
        background-color: #f7f5ed;
    }

    /* Style the Sign Up button */
    div.stButton > button {
        background-color: #ffe9a5;
        color: black;
        border: none;
        padding: 0.75em 2em; /* Bigger button */
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 100%; /* Make button fill the column */
    }
    div.stButton > button:hover {
        background-color: #ffd96b;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Logo at Top Left ----
col_logo, col_empty = st.columns([1, 5])

with col_logo:
    st.image("pictures/logo_small.png", width=100)  # Top left logo

with col_empty:
    st.empty()

# ---- Sidebar ----
with st.sidebar:
    st.header("BridgeSign")
    st.markdown("**Empowering Communication**")

# ---- Main Title ----
st.markdown(
    """
    <h1 style='text-align: center; color: #0277b5;'>The AI Sign Language Trainer for Healthcare</h1>
    <h4 style='text-align: center; color: #2aaaff;'>An interactive platform helping professionals build essential ASL skills through AI-driven practice.</h4>
    <h1 style='text-align: center; color: #0277b5;'>The AI Sign Language Trainer for Healthcare</h1>
    <h4 style='text-align: center; color: #2aaaff;'>An interactive platform helping professionals build essential ASL skills through AI-driven practice.</h4>
    """,
    unsafe_allow_html=True
)

# ---- Add vertical space before the Sign Up button ----
st.markdown("<br><br>", unsafe_allow_html=True)

# ---- Add vertical space before the Sign Up button ----
st.markdown("<br><br>", unsafe_allow_html=True)

# ---- Centered Call to Action: SIGN UP Button ----
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.empty()
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.empty()

with col2:
    if st.button("Sign Up", key="signup_button"):
        st.switch_page("pages/sign_up.py")
with col2:
    if st.button("Sign Up", key="signup_button"):
        st.switch_page("pages/sign_up.py")

with col3:
    st.empty()

# ---- Big Logo at the Bottom ----
st.markdown("<br><br>", unsafe_allow_html=True)

st.image("pictures/logo_big.png", use_column_width=True)

# ---- Footer (extra space at bottom) ----
# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
