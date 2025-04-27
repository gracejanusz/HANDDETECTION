import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import subprocess



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


def download_signsavvy_video(sign_word):
    videos_dir = "Videos"
    os.makedirs(videos_dir, exist_ok=True)

    video_path = os.path.join(videos_dir, f"{sign_word}.mp4")
    if os.path.exists(video_path):
        return video_path, False

    base_url = f"https://www.signingsavvy.com/search/{sign_word}"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    source_tag = soup.find('source', {'type': 'video/mp4'})

    video_url = None
    is_fingerspelled = False

    if source_tag and source_tag.has_attr('src'):
        video_url = source_tag['src']
        if not video_url.startswith('http'):
            video_url = "https://www.signingsavvy.com" + video_url
    else:
        video_path = build_fingerspelled_video_ffmpeg(sign_word.upper())
        if video_path:
            return video_path, True
        else:
            return None, False

    if video_url:
        video_data = requests.get(video_url).content
        with open(video_path, "wb") as f:
            f.write(video_data)
        return video_path, False
    else:
        return None, False

# Streamlit App
st.title("Signing Savvy Video Fetcher + Smart Fingerspell Fallback")

sign_word = st.text_input("Enter a sign word to download and display:")

if st.button("Fetch Video"):
    if sign_word:
        video_path, is_fingerspelled = download_signsavvy_video(sign_word.strip())

        if video_path:
            if is_fingerspelled:
                st.info("Fingerspelled version shown (constructed manually from letter videos).")
            else:
                st.success("Regular sign found and shown.")
            
            st.video(video_path)
        else:
            st.error("No video found or constructed for that sign.")
    else:
        st.warning("Please enter a word.")