import os
import requests
from bs4 import BeautifulSoup

def download_signsavvy_video(sign_word):
    videos_dir = "Videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    search_url = f"https://www.signingsavvy.com/search/{sign_word}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching search page for {sign_word}: {e}")
        return None, False

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # --- Primary attempt: find direct <source> ---
    source_tag = soup.find('source', {'type': 'video/mp4'})
    
    if source_tag and source_tag.has_attr('src'):
        video_url = source_tag['src']
        if not video_url.startswith('http'):
            video_url = "https://www.signingsavvy.com" + video_url

        try:
            video_data = requests.get(video_url).content
            video_path = os.path.join(videos_dir, f"{sign_word}.mp4")
            with open(video_path, "wb") as f:
                f.write(video_data)
            return video_path, False  # False = not fingerspelled
        except requests.RequestException as e:
            print(f"Error downloading video for {sign_word}: {e}")
            return None, False

    else:
        print("Direct video not found, attempting fallback...")
        
        # --- Fallback attempt: get first link in search_results ---
        first_link = soup.select_one('.search_results a')
        if first_link and first_link.has_attr('href'):
            fallback_url = first_link['href']
            if not fallback_url.startswith('http'):
                fallback_url = "https://www.signingsavvy.com/" + fallback_url.lstrip('/')

            try:
                fallback_response = requests.get(fallback_url)
                fallback_response.raise_for_status()
                fallback_soup = BeautifulSoup(fallback_response.text, 'html.parser')
                fallback_source_tag = fallback_soup.find('source', {'type': 'video/mp4'})

                if fallback_source_tag and fallback_source_tag.has_attr('src'):
                    video_url = fallback_source_tag['src']
                    if not video_url.startswith('http'):
                        video_url = "https://www.signingsavvy.com" + video_url

                    video_data = requests.get(video_url).content
                    video_path = os.path.join(videos_dir, f"{sign_word}.mp4")
                    with open(video_path, "wb") as f:
                        f.write(video_data)
                    return video_path, False
                else:
                    print(f"No video found on fallback page for {sign_word}.")
                    return None, False
            except requests.RequestException as e:
                print(f"Error during fallback download for {sign_word}: {e}")
                return None, False
        else:
            print(f"No fallback link found for {sign_word}.")
            return None, False
