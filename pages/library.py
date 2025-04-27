import streamlit as st
import os
from stickyhelper import st_fixed_container

# ---- Set page config ----
st.set_page_config(page_title="Library | BridgeSign", page_icon="📚", layout="wide")

# ---- Custom CSS Styling ----
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #f7f5ed;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #42b3ff !important;
        font-family: 'Georgia', serif !important;
        text-align: center;
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
        margin-bottom: 1em;
    }
    div.stButton > button:hover {
        background-color: #4dbbf4;
        color: black;
    }
    .welcome-text {
        color: #f8c434;
        font-size: 18px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .available-lessons {
        text-align: center;
        font-size: 24px;
        color: #0277b5;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st_fixed_container(mode="fixed", position="top", border=True, horizontal_position="right", key="top_right"):
    if st.button("Sign Up", key="signup_button"):
        st.switch_page("pages/sign_up.py")

    if st.button("Login", key="login_button"):
        st.switch_page("pages/log_in.py")

with st_fixed_container(mode="fixed", position="top", border=True, horizontal_position="left", key="top_left"):
    if st.button("Resources", key="signup_button1"):
        st.switch_page("pages/sign_up.py")

    if st.button("About Us", key="login_button2"):
        st.switch_page("pages/log_in.py")



col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col1:
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)

with col2:
    # ---- Library Page Content ----

    # Title
    st.markdown("<h1>Learning Library</h1>", unsafe_allow_html=True)

    # Welcome paragraph
    st.markdown(
        """
        <div class="welcome-text">
            Welcome to your BridgeSign learning library! Here you can access interactive lessons
            designed to help you build foundational ASL skills.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("--------")

    # ---- Available Lessons Section ----
    st.markdown("<div class='available-lessons'>Available Lessons</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #ccc; width: 80%; margin: auto;'>", unsafe_allow_html=True)

    # ---- Space before buttons ----
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Centered lesson buttons ----
    if st.button("📖 Learning the Alphabet", key="alphabet_lesson"):
        st.switch_page("pages/alphabet.py")

    if st.button("🗣️ Talk with an Avatar!", key="talk_with_avatar"):
        st.switch_page("pages/lesson_everyday.py")

    if st.button("🏃‍♂️ Verbs", key="verbs_lesson"):
        st.switch_page("pages/lesson_verbs.py")

    st.markdown("<br><br>", unsafe_allow_html=True)

with col3:
    st.image("pictures/minihands4.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)

# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
