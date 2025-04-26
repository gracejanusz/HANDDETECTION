import os
import requests
from bs4 import BeautifulSoup

def download_signsavvy_video(sign_word):
    # Create the Videos directory if it doesn't exist
    videos_dir = "Videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    search_url = f"https://www.signingsavvy.com/search/{sign_word}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Look for the <source> tag inside a <video> element
    source_tag = soup.find('source', {'type': 'video/mp4'})
    
    if source_tag and source_tag.has_attr('src'):
        video_url = source_tag['src']
        if not video_url.startswith('http'):
            video_url = "https://www.signingsavvy.com" + video_url
        
        print(f"Downloading {sign_word} video from: {video_url}")
        video_data = requests.get(video_url).content
        
        # Save the video inside the Videos folder
        video_path = os.path.join(videos_dir, f"{sign_word}.mp4")
        with open(video_path, "wb") as f:
            f.write(video_data)
        print(f"Download complete! Video saved as: {video_path}")
    else:
        print("No video found for that sign.")

# Ask the user for input
if __name__ == "__main__":
    sign_word = input("Enter the sign word you want to download: ").strip()
    download_signsavvy_video(sign_word)
