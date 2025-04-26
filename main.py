import streamlit as st
from PIL import Image
import firebase_admin
from firebase_admin import auth
import base64
import io


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_image_html(image_path, width=575):
    img_base64 = get_base64_of_bin_file(image_path)
    html_code = f"<img id='logo' class='logo-start' src='data:image/png;base64,{img_base64}' width='{width}'>"
    return html_code

# ---- Set page config ----
st.set_page_config(page_title="BridgeSign", page_icon="🧏‍♀️", layout="wide")

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
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        transition: all 0.3s ease-in-out;
    }
    @keyframes bounce {
        0%   { transform: scale(1); }
        50%  { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    /* Logo centering when page loads */
    img.logo-start {
        display: block;
        margin-left: auto;
        margin-right: auto;
        transition: all 0.3s ease-in-out;
    }
    .sticky {
        position: fixed;
        top: 0px;
        left: 20px;
        width: 120px; /* smaller width when scrolled */
        margin: 0;
        animation: bounce 0.4s ease;
        z-index: 9999;
    }
    </style>

    <script>
    window.onscroll = function() {myFunction()};
    function myFunction() {
        var logo = document.getElementById('logo');
        if (document.documentElement.scrollTop > 50) {
            logo.classList.add('sticky');
        } else {
           logo.classList.remove('sticky');
        }
    }
    </script>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar ----
with st.sidebar:
    st.header("BridgeSign")
    st.markdown("**Empowering Communication**")

# ---- Add vertical space before buttons ----
st.markdown("<br><br>", unsafe_allow_html=True)

# ---- Centered Call to Action: SIGN UP and LOGIN Buttons ----
col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col1:
    st.image("pictures/minihands1.png", use_container_width=True)
    st.image("pictures/minihands2.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)


with col2:
    st.markdown(get_image_html("pictures/big_logo.png"), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)

    # Sign Up Button
    if st.button("Sign Up", key="signup_button"):
        st.switch_page("pages/sign_up.py")  # Navigates to pages/sign_up.py

    # Login Button
    if st.button("Login", key="login_button"):
        st.switch_page("pages/log_in.py")  # Navigates to pages/login_page.py

with col3:
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)
    st.image("pictures/minihands1.png", use_container_width=True)
    st.image("pictures/minihands2.png", use_container_width=True)



# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)

