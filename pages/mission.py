# pages/mission.py

import streamlit as st

# ---- Page Config ----
st.set_page_config(page_title="Our Mission", page_icon="✨", layout="wide")

# ---- Custom CSS Styling ----
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f5ed;
        animation: fadeInAnimation ease 1s;
        animation-iteration-count: 1;
        animation-fill-mode: forwards;
    }

    @keyframes fadeInAnimation {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }

    h1, h2, h3, h4, h5, h6 {
        color: #42b3ff !important;
        text-align: center;
    }
    .stApp p {
        color: #0277b5;
        font-size: 18px;
        line-height: 1.6;
    }
    .clickable-logo {
        position: absolute;
        top: 10px;
        left: 10px;
        cursor: pointer;
        transform-origin: center center;
    }
    .clickable-logo:hover {
        transform: scale(0.95);
        transition: transform 0.2s ease;
    }
    /* Custom Button Styling */
    div.stButton > button {
        background-color: #f8c434 !important;
        color: black !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1.2em !important;
        font-weight: bold !important;
        padding: 0.75em 1.5em !important;
        margin-top: 20px !important;
        width: 80% !important; /* Wider button */
        display: block !important;
        margin-left: 10% !important; /* Slightly left (not centered) */
    }
    div.stButton > button:hover {
        background-color: #ffd700 !important; /* Lighter gold on hover */
        color: black !important;
    }
    </style>
    <script>
        window.scrollTo({top: 0, behavior: 'smooth'});
    </script>
    """,
    unsafe_allow_html=True
)

# ---- Layout ----
col1, col2, col3 = st.columns([1.5, 3, 1.5])

with col1:
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)

with col2:
    # ---- Page Content ----
    st.markdown("<h1>Our Mission</h1>", unsafe_allow_html=True)

    st.write("""
    We are a B2B platform designed to help service providers build essential ASL skills through interactive, AI-driven practice.
    By equipping professionals in healthcare and other public-facing fields with foundational ASL literacy, we address a critical communication gap,
    allowing institutions to better serve their Deaf and Hard of Hearing (D/HH) patients.
    """)

    st.markdown("---")

    st.markdown("<h2>Our Approach to ASL and AI</h2>", unsafe_allow_html=True)

    st.write("""
    We recognize that conversations around ASL education and AI can bring up important and valid concerns, so we would like to provide more detailed information about our approach.

    **First**, we are not aiming to replace D/HH professionals, including interpreters, whose work is vital and irreplaceable.
    Our platform is designed to provide *basic ASL literacy* to service providers, particularly in healthcare and similar settings,
    where communication difficulties with D/HH individuals are still far too common.

    **Second**, we acknowledge that ASL is an expressive language where emotion, facial expressions, and cultural context carry deep meaning.
    No AI can fully replicate that. Our goal is not to teach fluency or mimic the full experience of human communication,
    but to provide a space for basic communication practice and foster more respectful and effective interactions.
    """)

    st.markdown("---")

    st.markdown("<h2>How We Are Different</h2>", unsafe_allow_html=True)

    st.write("""
    Most tools in this space are built to translate Deaf individuals into the hearing world (e.g., voice-to-text or sign-to-speech).
    We are flipping that approach — building a tool to help *institutions* adapt to and respect ASL as a primary language,
    rather than expecting the D/HH community to adapt.

    Our mission is grounded in respect, accessibility, and partnership.
    """)

    st.markdown("---")

    st.caption("Made by Grace Janusz, Nusret Efe Ucer, Defne Meric Erdogan, and Elena Loucks")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- Go Back to Main Page Button ----
    if st.button("Go to Main Page", key="back_home_button_mission", use_container_width=True):
        st.switch_page("main.py")  # Make sure you have streamlit-extras or adjust if needed

with col3:
    st.image("pictures/minihands4.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands3.png", use_container_width=True)
    st.image("pictures/minihands4.png", use_container_width=True)

# ---- Footer (extra space at bottom) ----
st.markdown("<br><br>", unsafe_allow_html=True)
