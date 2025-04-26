import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth

# ---- Initialize Firebase (only once) ----
if not firebase_admin._apps:
    cred = credentials.Certificate('handsin-e15d2-7fe9c1f743a4.json')
    firebase_admin.initialize_app(cred)

# ---- Set page config ----
st.set_page_config(page_title="Sign Up | BridgeSign", page_icon="🧏‍♀️", layout="centered")

# ---- Sign Up Form ----
st.title("Create Your BridgeSign Account")

email = st.text_input("Email")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

signup_button = st.button("Sign Up")

if signup_button:
    if password != confirm_password:
        st.error("Passwords do not match.")
    elif len(password) < 6:
        st.error("Password must be at least 6 characters long.")
    else:
        try:
            user = auth.create_user(
                email=email,
                password=password
            )
            st.success("Account created successfully! 🎉 You can now log in.")
            st.balloons()

            # Optional: Go back to Home
            if st.button("Go to Home"):
                st.switch_page("main.py")
        except Exception as e:
            st.error(f"Signup failed: {e}")

# ---- Back to Home Button (in case they change their mind) ----
if st.button("Back to Home", key="back_home_button"):
    st.switch_page("main.py")
