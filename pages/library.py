import streamlit as st
import os

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
    .lesson-button {
        text-align: left;
        margin-left: 10%;
    }
    .lesson-button button {
        background-color: #ffe9a5;
        color: black;
        border: none;
        padding: 0.75em 2em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 300px;
        margin-top: 1em;
        display: block;
    }
    .lesson-button button:hover {
        background-color: #ffd96b;
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
        text-align: left;
        font-size: 24px;
        color: #0277b5;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 5px;
        margin-left: 10%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col1:
    st.image("pictures/minihands1.png", use_container_width=True)
    st.image("pictures/minihands2.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)

with col2:
    # ---- Library Page Content ----

    # Title
    st.markdown("<h1>Learning Library</h1>", unsafe_allow_html=True)

    # Yellow welcome paragraph
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

    st.markdown("<hr style='border: 1px solid #ccc; margin-left: 10%; width: 80%;'>", unsafe_allow_html=True)

    # ---- Space before buttons ----
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Left-aligned lesson buttons ----
    st.markdown("<div class='lesson-button'>", unsafe_allow_html=True)

    # Alphabet Course button
    if st.button("📖 Learning the Alphabet", key="alphabet_lesson", help="Click to start learning the ASL alphabet"):
        # Hacky way to run model26.py when button is clicked
        os.system("streamlit run model26.py")
        st.stop()

    # Other placeholder buttons
    if st.button("🗣️ Everyday Talk", key="everyday_talk_lesson", help="Click to learn everyday phrases"):
        st.switch_page("pages/lesson_everyday.py")  # Placeholder

    if st.button("🏃‍♂️ Verbs", key="verbs_lesson", help="Click to learn common action verbs"):
        st.switch_page("pages/lesson_verbs.py")  # Placeholder

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

with col3:
    st.image("pictures/minihands4.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands2.png", use_container_width=True)
    st.image("pictures/minihands1.png", use_container_width=True)

# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)