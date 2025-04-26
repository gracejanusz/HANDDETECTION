from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def setup_driver():
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    return driver

def search_word_and_get_video(driver, word):
    driver.get("https://www.spreadthesign.com/en.us/search/")

    try:
        # Wait until the input with id 'searchWord' is present
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "searchWord"))
        )
        search_box.clear()
        search_box.send_keys(word)
        search_box.send_keys(Keys.RETURN)

        # Wait for a video element to appear
        video_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "video"))
        )
        video_url = video_element.get_attribute("src")
        return video_url

    except Exception as e:
        print(f"Could not find video for {word}: {e}")
        return None

def gloss_to_videos(gloss_text):
    driver = setup_driver()
    words = gloss_text.strip().upper().split()
    video_links = {}

    for word in words:
        video_url = search_word_and_get_video(driver, word)
        if video_url:
            video_links[word] = video_url
        else:
            video_links[word] = "Not found"

    driver.quit()
    return video_links

if __name__ == "__main__":
    gloss_input = input("Enter ASL gloss text (e.g., 'BOY EAT APPLE'): ")
    videos = gloss_to_videos(gloss_input)

    print("\nVideo Links:")
    for word, link in videos.items():
        print(f"{word}: {link}")
