import streamlit as st
import os
import subprocess
import sys
import glob # Import glob for file searching
import google.generativeai as genai # Import Gemini API
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gloss_to_english import english_to_gloss
from get_video import download_signsavvy_video
from English_Video import build_fingerspelled_video_ffmpeg, concatenate_videos_ffmpeg
from mr_robot import generate_avatar
from config import GEMINI_API_KEY

# Configure Gemini API (replace with your actual API key or use st.secrets)
# It's recommended to use st.secrets for production
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')


st.title("Avatar Generation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Video Feed")
    user_video_feed = st.camera_input("Live Feed", label_visibility="hidden")

# The avatar video will be displayed in col2 later

english_phrase = st.text_input("Enter an English phrase to generate an avatar video:")

if st.button("Generate Avatar Video"):
    if english_phrase:

        try:
            # Call Gemini API to get a response
            response = model.generate_content(english_phrase.strip())
            gemini_response_text = response.text.strip()
            if english_phrase == 'Hello':
                gemini_response_text = 'My Name is Andy what is your name'

            # Use the Gemini response for gloss translation and video generation
            gloss_phrase = english_to_gloss(gemini_response_text)
            gloss_filename_base = gloss_phrase.replace(" ", "_")

            # Define the directory where avatar videos are saved
            avatar_output_dir = os.path.join("Videos", "avatar_output")
            expected_filename_pattern = os.path.join(avatar_output_dir, f"{gloss_filename_base}*initial.mp4")

            # Check if a matching file already exists
            existing_videos = glob.glob(expected_filename_pattern)

            if existing_videos:
                # Display the first matching video found
                with col2:
                    st.video(existing_videos[0])
                st.success("Displayed existing AI Avatar video.")
                st.stop()
            else:

                # Split gloss into words
                gloss_words = gloss_phrase.split()

                video_paths = []
                for word in gloss_words:
                    # Define potential video path based on word
                    word_video_path = os.path.join("Videos", f"{word}.mp4")

                    if os.path.exists(word_video_path):
                        video_paths.append(word_video_path)
                    else:
                        video_path, is_sign_video = download_signsavvy_video(word)
                        is_fingerspelled = False
                        if video_path is None:
                            video_path = build_fingerspelled_video_ffmpeg(word)
                            if video_path:
                                is_fingerspelled = True

                        if video_path:
                            video_paths.append(video_path)
                            if is_fingerspelled:
                                pass
                            else:
                                pass
                        else:
                            pass

                if video_paths:
                    output_video_path = os.path.join("Videos", gloss_filename_base + "_initial.mp4") # Use underscores and add _initial
                    concatenated_video = concatenate_videos_ffmpeg(video_paths, output_video_path)

                    if concatenated_video:
                        st.info("Creating Ai video")

                        # Call mr_robot.py function with the generated video path
                        final_video_path = generate_avatar(concatenated_video)

                        if final_video_path:
                            st.success("AI Avatar video generated:")
                            with col2:
                                st.video(final_video_path)
                        else:
                            st.error("Failed to generate or retrieve the AI Avatar video.")

                    else:
                        st.error("Failed to concatenate initial videos.")
                else:
                    st.error("No initial videos could be generated for the phrase.")

        except Exception as e:
            st.error(f"An error occurred during Gemini API call or subsequent processing: {e}")

    else:
        st.warning("Please enter a phrase.")