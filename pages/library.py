# pages/library.py

import streamlit as st

# ---- Set page config ----
st.set_page_config(page_title="Library | BridgeSign", page_icon="📚", layout="centered")

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
    .lesson-button button {
        background-color: #ffe9a5;
        color: black;
        border: none;
        padding: 0.75em 2em;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        width: 50%;
        margin: 1em auto;
        display: block;
    }
    .lesson-button button:hover {
        background-color: #ffd96b;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Library Page Content ----

# Title
st.markdown("<h1>Learning Library</h1>", unsafe_allow_html=True)

st.write("""
Welcome to your BridgeSign learning library! Here you can access interactive lessons 
designed to help you build foundational ASL skills.
""")

st.markdown("---")

# ---- Alphabet Lesson Button ----
st.markdown("<h2>Available Lessons</h2>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Center the lesson button
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    st.empty()
with col2:
    with st.container():
        # Lesson Button (Styled)
        if st.button("📖 Learning the Alphabet", key="alphabet_lesson", help="Click to start learning the ASL alphabet"):
            st.switch_page("pages/lesson_alphabet.py")  # ➡️ You will create this lesson page next
with col3:
    st.empty()

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.caption("Keep learning, one sign at a time! 🤟")
