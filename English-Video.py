# English to asl video
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import subprocess
from gloss_to_english import english_to_gloss
from get_video import download_signsavvy_video



def build_fingerspelled_video_ffmpeg(word):
    videos_dir = "Videos"
    temp_list_file = os.path.join(videos_dir, "file_list.txt")
    output_path = os.path.join(videos_dir, f"{word}.mp4")

    letter_videos = []
    i = 0
    while i < len(word):
        if i+1 < len(word) and word[i] == word[i+1]:
            double_letter = word[i]*2
            double_path = os.path.join(videos_dir, f"{double_letter}.mp4")
            if os.path.exists(double_path):
                letter_videos.append(double_path)
                i += 2
                continue

        letter_path = os.path.join(videos_dir, f"{word[i]}.mp4")
        if os.path.exists(letter_path):
            letter_videos.append(letter_path)
        else:
            print(f"Missing video for letter: {word[i]}")
        i += 1

    if not letter_videos:
        return None

    # Create a temporary file list
    with open(temp_list_file, 'w') as f:
        for video in letter_videos:
            f.write(f"file '{os.path.abspath(video)}'\n")

    # Run ffmpeg concat
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', temp_list_file,
        '-c', 'copy',
        output_path
    ]

    subprocess.run(command)

    # Clean up temp list
    os.remove(temp_list_file)

    return output_path


def concatenate_videos_ffmpeg(video_paths, output_path):
    if not video_paths:
        return None

    command = []
    for path in video_paths:
        command.extend(['-i', os.path.abspath(path)])

    # ONLY reference video streams [i:v:0] (no audio)
    filter_complex = ''.join(f'[{i}:v:0]' for i in range(len(video_paths)))
    filter_complex += f'concat=n={len(video_paths)}:v=1:a=0[outv]'

    ffmpeg_command = [
        'ffmpeg',
        *command,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-y',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg run: {e}")
        return None


# Streamlit App
st.title("Signing Savvy Video Fetcher + Smart Fingerspell Fallback")

english_phrase = st.text_input("Enter an English phrase to translate and display as video:")

if st.button("Generate Video"):
    if english_phrase:
        # Translate English to Gloss
        gloss_phrase = english_to_gloss(english_phrase.strip())
        print(gloss_phrase)
        st.info(f"Translated Gloss: {gloss_phrase}")

        # Split gloss into words
        gloss_words = gloss_phrase.split()

        video_paths = []
        for word in gloss_words:
            video_path, is_sign_video = download_signsavvy_video(word) # Renamed is_fingerspelled to is_sign_video for clarity from get_video.py's perspective
            is_fingerspelled = False # Initialize flag
            if video_path is None:
                # If sign video not found, try fingerspelling
                video_path = build_fingerspelled_video_ffmpeg(word)
                if video_path:
                    is_fingerspelled = True

            if video_path:
                video_paths.append(video_path)
                if is_fingerspelled:
                    st.info(f"Using fingerspelled video for '{word}'.")
                else:
                    st.success(f"Using sign video for '{word}'.")
            else:
                st.warning(f"Could not find or create video for '{word}'. Skipping.")

        # Now, concatenate the videos
        if video_paths:
            output_video_path = os.path.join("Videos", gloss_phrase + ".mp4")
            concatenated_video = concatenate_videos_ffmpeg(video_paths, output_video_path)

            if concatenated_video:
                st.success("Combined video generated:")
                st.video(concatenated_video)
            else:
                st.error("Failed to concatenate videos.")
        else:
            st.error("No videos could be generated for the phrase.")

    else:
        st.warning("Please enter a phrase.")