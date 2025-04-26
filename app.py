# app.py

import streamlit as st

# Set the page configuration (optional but nice)
st.set_page_config(
    page_title="Hand Detection App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.markdown(
    "<h1 style='text-align: center; color: teal;'>🤖 Welcome to Hand Detection App</h1>",
    unsafe_allow_html=True
)


# Small description
st.write("""
Welcome! Upload an image or start your camera to detect hands using MediaPipe + OpenCV.  
Built with ❤️ and Python 3.9.
""")

# Sidebar for user options
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the mode", ["Upload Image", "Live Camera"])

# Main functionality
if app_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # TODO: Run your hand detection function here
        st.success("✅ Ready to detect hands!")
        # Example: result_image = detect_hands(uploaded_file)
        # st.image(result_image, caption="Processed Image", use_column_width=True)

elif app_mode == "Live Camera":
    st.info("📷 Live camera hand detection coming soon!")
    st.write("👉 (You can integrate OpenCV's webcam stream here.)")

# Footer
st.markdown("---")
st.markdown("Made by [Your Name] ✨")