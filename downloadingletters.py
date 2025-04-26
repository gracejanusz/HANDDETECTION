import os
import requests
from bs4 import BeautifulSoup
import time

def download_signsavvy_video(sign_word):
    # Create the Videos directory if it doesn't exist
    videos_dir = "Videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    # First, try normal sign
    search_url = f"https://www.signingsavvy.com/search/{sign_word}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    source_tag = soup.find('source', {'type': 'video/mp4'})
    
    video_url = None

    if source_tag and source_tag.has_attr('src'):
        video_url = source_tag['src']
        if not video_url.startswith('http'):
            video_url = "https://www.signingsavvy.com" + video_url
    else:
        # If no normal sign, fallback to fingerspelled page
        fingerspell_url = f"https://www.signingsavvy.com/sign/{sign_word.upper()}/0/fingerspell"
        response_fingerspell = requests.get(fingerspell_url)
        soup_fingerspell = BeautifulSoup(response_fingerspell.text, 'html.parser')
        source_tag = soup_fingerspell.find('source', {'type': 'video/mp4'})
        
        if source_tag and source_tag.has_attr('src'):
            video_url = source_tag['src']
            if not video_url.startswith('http'):
                video_url = "https://www.signingsavvy.com" + video_url
    
    if video_url:
        # Download the video
        video_path = os.path.join(videos_dir, f"{sign_word}.mp4")
        video_data = requests.get(video_url).content
        with open(video_path, "wb") as f:
            f.write(video_data)
        print(f"Downloaded: {sign_word}")
    else:
        print(f"No video found for: {sign_word}")

def batch_download_letters_and_vowels():
    # Letters A-Z
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Double vowels
    double_vowels = ["AA", "EE", "II", "OO", "UU"]

    # Combine both lists
    words_to_download = letters + double_vowels

    print(f"Downloading {len(words_to_download)} signs...")

    for word in words_to_download:
        try:
            download_signsavvy_video(word)
            time.sleep(1)  # Be polite: wait 1 second between requests
        except Exception as e:
            print(f"Error downloading {word}: {e}")

if __name__ == "__main__":
    # Option 1: manual input
    choice = input("Enter '1' to download a single word, '2' to batch download A-Z and double vowels: ").strip()

    if choice == "1":
        sign_word = input("Enter the sign word you want to download: ").strip()
        download_signsavvy_video(sign_word)
    elif choice == "2":
        batch_download_letters_and_vowels()
    else:
        print("Invalid choice. Exiting.")
