# pages/mission.py

import streamlit as st

st.set_page_config(page_title="Our Mission", page_icon=":sparkles:")

# ---- Custom CSS for background and text colors ----
st.markdown(
    """
    <style>
    body {
        background-color: #f7f5ed;
        color: #0277b5;
    }
    .stApp {
        background-color: #f7f5ed;
        color: #0277b5;
    }
    h1 {
        color: #0277b5;
    }
    .stSubheader {
        color: #f8c434 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Mission Content ----
st.title("Our Mission")
st.write("""
We are a B2B platform designed to help service providers build essential ASL skills through interactive, AI-driven practice. 
By equipping professionals in healthcare and other public-facing fields with foundational ASL literacy, we address a critical communication gap, 
allowing institutions to better serve their Deaf and Hard of Hearing (D/HH) patients.

---
""")

st.subheader("Our Approach to ASL and AI")
st.write("""
We recognize that conversations around ASL education and AI can bring up important and valid concerns, so we would like to provide more detailed information about our approach.

**First**, we are not aiming to replace D/HH professionals, including interpreters, whose work is vital and irreplaceable. 
Our platform is designed to provide *basic ASL literacy* to service providers, particularly in healthcare and similar settings, 
where communication difficulties with D/HH individuals are still far too common.

**Second**, we acknowledge that ASL is an expressive language where emotion, facial expressions, and cultural context carry deep meaning. 
No AI can fully replicate that. Our goal is not to teach fluency or mimic the full experience of human communication, 
but to provide a space for basic communication practice and foster more respectful and effective interactions.

---
""")

st.subheader("How We Are Different")
st.write("""
Most tools in this space are built to translate Deaf individuals into the hearing world (e.g., voice-to-text or sign-to-speech). 
We are flipping that approach — building a tool to help *institutions* adapt to and respect ASL as a primary language, 
rather than expecting the D/HH community to adapt.

Our mission is grounded in respect, accessibility, and partnership.
""")

st.markdown("---")
st.caption("Made with ❤️ for the D/HH community")

