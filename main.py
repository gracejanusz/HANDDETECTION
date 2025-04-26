import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="BridgeSign", page_icon="🧏‍♀️", layout="centered")

# ---- CUSTOM CSS to make background white ----
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Big Logo at the Top ----
st.image("pictures/big_logo.png", use_container_width=True)
<<<<<<< HEAD

# ---- Logo at Top Left ----
col_logo, col_empty = st.columns([1, 5])

with col_logo:
    st.image("pictures/logo_small.png", width=100)  # Top left logo

with col_empty:
    st.empty()
=======
>>>>>>> 6858355a (color changes)

# ---- Sidebar ----
with st.sidebar:
    st.header("BridgeSign")
    st.markdown("**Empowering Communication**")

# ---- Main Title ----
st.markdown(
    """
    <h1 style='text-align: center; color: #20522e;'>The AI Sign Language Trainer for Healthcare</h1>
    <h4 style='text-align: center; color: #444;'>An interactive platform helping professionals build essential ASL skills through AI-driven practice.</h4>
    """,
    unsafe_allow_html=True
)

# ---- Centered Call to Action: SIGN UP Button ----
st.markdown("<div style='text-align: center; padding: 100;'>", unsafe_allow_html=True)

if st.button("Sign Up", key="signup_button"):
    st.switch_page("pages/sign_up.py")

with col3:
    st.empty()

<<<<<<< HEAD
# ---- Trust Section ----
st.markdown(
    """
    <div style='text-align: center; margin-top: 30px; color: #888;'>
        Backed by <img src='https://upload.wikimedia.org/wikipedia/commons/6/69/Y_Combinator_logo.svg' width='90'>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Footer ----
=======
# ---- Footer (extra space at bottom) ----
>>>>>>> 6858355a (color changes)
st.markdown("<br><br>", unsafe_allow_html=True)
