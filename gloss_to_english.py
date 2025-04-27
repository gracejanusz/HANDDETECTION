import google.generativeai as genai
import os
from dotenv import load_dotenv
from config import GEMINI_API_KEY  # Import API key from config.py

# Setup Gemini API Key

genai.configure(api_key=GEMINI_API_KEY)

def gloss_to_english(gloss_text):
    """Converts ASL gloss to natural English sentences."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    convo = model.start_chat()
    prompt = f"Convert the following ASL gloss into natural English sentences:\n\nGloss: {gloss_text}\nEnglish:"
    try:
        convo.send_message(prompt)
        english_text = convo.last.text.strip()
        print(english_text)
        return english_text
    except Exception as e:
        print(f"An error occurred during gloss to English conversion: {e}")
        return "Error during translation."

def english_to_gloss(english_text):
    """Converts English sentences into ASL gloss (keywords, no extra words)."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    convo = model.start_chat()
    prompt = f"Convert the following English sentence into ASL gloss (keywords, no extra words):\n\nEnglish: {english_text}\nGloss:"
    try:
        convo.send_message(prompt)
        gloss_text = convo.last.text.strip()
        # Check if the response starts with "GLOSS: " and remove it
        if gloss_text.startswith("GLOSS: "):
            gloss_text = gloss_text[len("GLOSS: "):].strip()
        # Ensure gloss is uppercase
        return gloss_text.upper()
    except Exception as e:
        print(f"An error occurred during English to gloss conversion: {e}")
        return "Error during translation."

if __name__ == "__main__":
    print("ASL Gloss <-> English Translator")
    print("1. Gloss to English")
    print("2. English to Gloss")

    while True:
        choice = input("Choose an option (1 or 2): ")
        if choice in ['1', '2']:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if choice == '1':
        gloss_input = input("Enter ASL Gloss: ")
        english_output = gloss_to_english(gloss_input)
        print("\nTranslated English:")
        print(english_output)
    else: # choice == '2'
        english_input = input("Enter English sentence: ")
        gloss_output = english_to_gloss(english_input)
        print("\nTranslated ASL Gloss:")
        print(gloss_output)