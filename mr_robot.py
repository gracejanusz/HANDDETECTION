from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# --- Credentials ---
EMAIL = "masterhacker1632@gmail.com"
PASSWORD = "Mstrhckr1632" # Consider using environment variables or a config file for credentials

def generate_avatar(file_name):
    # Set up the Chrome driver using webdriver-manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.maximize_window() # Maximize window for better element visibility

    # The URL to open
    url = "https://www.deepmotion.com/animate-3d"

    # Open the webpage
    driver.get(url)

    print(f"Opened page: {driver.title}")

    # Wait for the "Sign in" link to be clickable and then click it
    try:
        sign_in_locator = (By.XPATH, "//div[contains(@class, 'cursor-pointer') and normalize-space(text())='Sign in']")
        sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(sign_in_locator)
        )
        sign_in_button.click()
        print("Clicked the 'Sign in' link.")

        # --- Wait for login form elements and fill them ---
        # Wait for email input to be visible
        email_input_locator = (By.ID, "username")
        email_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located(email_input_locator)
        )
        email_input.send_keys(EMAIL)
        print(f"Entered email: {EMAIL}")

        # Find password input (assuming it becomes visible shortly after email)
        password_input_locator = (By.ID, "password")
        password_input = driver.find_element(*password_input_locator) # Find immediately after email is entered
        password_input.send_keys(PASSWORD)
        print("Entered password.")

        # Click the final "Sign In" button
        final_sign_in_locator = (By.XPATH, "//div[contains(@class, 'launch-button') and contains(@class, 'signIn')]")
        final_sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(final_sign_in_locator)
        )
        final_sign_in_button.click()
        print("Clicked the final 'Sign In' button.")
        time.sleep(2)

        # --- Redirect to the portal page ---
        portal_url = "https://portal.deepmotion.com/"
        print(f"Redirecting to: {portal_url}")
        driver.get(portal_url)
        print(f"Navigated to portal page: {driver.title}")

        # Wait for and click the "Create" button/span
        create_button_locator = (By.XPATH, "//span[contains(@class, 'MuiListItemText-primary') and normalize-space(text())='Create']")
        print(f"Waiting for Create button with locator: {create_button_locator}") # Debugging print
        create_button = WebDriverWait(driver, 20).until( # Wait for the portal page elements to load
            EC.element_to_be_clickable(create_button_locator)
        )
        print("Create button found and clickable.") # Debugging print
        create_button.click()
        print("Clicked the 'Create' button.")
        time.sleep(2) # Wait for the page to load

        # --- Click the Edit Icon Button ---
        # Locator targets the button containing the edit icon, using aria-label
        edit_button_locator = (By.XPATH, "//button[@aria-label='Change the model']")
        print(f"Waiting for Edit icon button with locator: {edit_button_locator}")
        edit_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(edit_button_locator) # Wait for presence for JS click
        )
        print("Edit icon button found.")
        driver.execute_script("arguments[0].click();", edit_button)
        print("Clicked the Edit icon button using JavaScript.")
        time.sleep(2) # Increase pause slightly for model options to load
        # --------------------------

        # --- Select the Character Model ---
        model_name = "avatar_1" # Updated model name
        model_locator = (By.XPATH, f"//div[contains(@class, 'ModelSelector-characters') and .//div[normalize-space(text())='{model_name}']]")
        print(f"Waiting for model '{model_name}' with locator: {model_locator}")
        model_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(model_locator) # Wait for presence for JS click
        )
        print(f"Model '{model_name}' found.")
        driver.execute_script("arguments[0].click();", model_element)
        print(f"Clicked model '{model_name}' using JavaScript.")
        time.sleep(1) # Short pause after selecting model
        # --------------------------

        # --- Click Apply and Close button ---
        apply_close_button_locator = (By.XPATH, "//button[.//span[normalize-space(text())='Apply and Close']]")
        # Alternative CSS: (By.CSS_SELECTOR, "button.dm-button.css-1qi5k2b")
        print(f"Waiting for 'Apply and Close' button with locator: {apply_close_button_locator}")
        apply_close_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(apply_close_button_locator) # Wait for presence for JS click
        )
        print("'Apply and Close' button found.")
        driver.execute_script("arguments[0].click();", apply_close_button)
        print("Clicked 'Apply and Close' button using JavaScript.")
        time.sleep(1) # Short pause after closing model selection
        # --------------------------

        # --- Upload File using the hidden input element ---
        # Define the absolute path to the file you want to upload
        file_to_upload = f"/Users/efeucer/Desktop/hacktech_2/HANDDETECTION/{file_name}.mp4"
        absolute_file_path = os.path.abspath(file_to_upload)
        print(f"Attempting to upload file: {absolute_file_path}")

        if not os.path.exists(absolute_file_path):
            print(f"ERROR: File not found at path: {absolute_file_path}")
        else:
            try:
                # Locate the hidden file input element by its ID
                file_input_locator = (By.ID, "fileInput")
                print(f"Waiting for file input element with locator: {file_input_locator}")

                # Wait for the input element to be present in the DOM
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(file_input_locator)
                )
                print("Hidden file input element found.")

                # Send the absolute file path to the input element
                file_input.send_keys(absolute_file_path)
                print(f"Sent file path '{absolute_file_path}' to the input element.")

                # Add a pause or wait for upload confirmation if necessary
                print("Waiting a few seconds for upload to potentially start...")
                time.sleep(5) # Adjust as needed

                # --- Toggle the "Hand Tracking" switch using a specific sibling relationship ---
                # Locate the checkbox input by finding the label's container and then the following switch input
                hand_tracking_switch_locator = (By.XPATH, "//div[contains(@class, 'is-flex') and ./div[normalize-space(text())='Hand Tracking']]/following-sibling::span[contains(@class, 'MuiSwitch-root')]//input[@type='checkbox']")
                # Explanation:
                # 1. Find the specific inner div that has 'is-flex' and DIRECTLY contains the 'Hand Tracking' div (using './div')
                # 2. Look for the 'span' element that is the immediate FOLLOWING SIBLING of that div AND contains the class 'MuiSwitch-root'
                # 3. Find the 'input' of type 'checkbox' within that specific sibling span

                print(f"Waiting for Hand Tracking switch input checkbox with locator: {hand_tracking_switch_locator}")

                # Wait for the checkbox input to be present
                hand_tracking_switch_input = WebDriverWait(driver, 10).until( # Increased wait time
                    EC.presence_of_element_located(hand_tracking_switch_locator)
                )
                print("Hand Tracking switch input checkbox found.")

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", hand_tracking_switch_input)
                print("Clicked the Hand Tracking switch input checkbox using JavaScript.")
                time.sleep(1)
                # --------------------------

                # --- Toggle the "Upper Body Only" switch using a specific sibling relationship ---
                # Locate the checkbox input by finding the label's container and then the following switch input
                upper_body_switch_locator = (By.XPATH, "//div[contains(@class, 'is-flex') and ./div[normalize-space(text())='Upper Body Only']]/following-sibling::span[contains(@class, 'MuiSwitch-root')]//input[@type='checkbox']")
                print(f"Waiting for Upper Body Only switch input checkbox with locator: {upper_body_switch_locator}")

                # Wait for the checkbox input to be present
                upper_body_switch_input = WebDriverWait(driver, 10).until( # Using 10s wait, adjust if needed
                    EC.presence_of_element_located(upper_body_switch_locator)
                )
                print("Upper Body Only switch input checkbox found.")

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", upper_body_switch_input)
                print("Clicked the Upper Body Only switch input checkbox using JavaScript.")
                time.sleep(1)
                # --------------------------

                # --- Click the "MP4 Render Settings" button ---
                mp4_settings_button_locator = (By.XPATH, "//button[normalize-space(text())='MP4 Render Settings']")
                print(f"Waiting for MP4 Render Settings button with locator: {mp4_settings_button_locator}")

                # Wait for the button to be present (clickable might fail due to interception)
                mp4_settings_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(mp4_settings_button_locator) # Changed to presence_of_element_located
                )
                print("MP4 Render Settings button found.")

                # Click using JavaScript to bypass potential interception
                driver.execute_script("arguments[0].click();", mp4_settings_button)
                print("Clicked the MP4 Render Settings button using JavaScript.")
                time.sleep(2) # Increase pause slightly for settings panel to appear

                # --- Toggle the "MP4 Output" switch within the MP4 Render Settings ---
                # Set the actual label text for the switch
                mp4_switch_label_text = "MP4 Output" # <-- Updated label text
                # Use the same following-sibling strategy as Hand Tracking/Upper Body
                mp4_switch_locator = (By.XPATH, f"//div[contains(@class, 'is-flex') and ./div[normalize-space(text())='{mp4_switch_label_text}']]/following-sibling::span[contains(@class, 'MuiSwitch-root')]//input[@type='checkbox']")
                # Explanation:
                # 1. Find the specific inner div that has 'is-flex' and DIRECTLY contains the label text ('MP4 Output')
                # 2. Look for the 'span' element that is the immediate FOLLOWING SIBLING of that div AND contains the class 'MuiSwitch-root'
                # 3. Find the 'input' of type 'checkbox' within that specific sibling span

                print(f"Waiting for MP4 settings switch ('{mp4_switch_label_text}') with locator: {mp4_switch_locator}")

                # Wait for the switch input to be present
                mp4_switch_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(mp4_switch_locator)
                )
                print(f"Found the MP4 settings switch input ('{mp4_switch_label_text}').")

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", mp4_switch_input)
                print(f"Clicked the MP4 settings switch ('{mp4_switch_label_text}') using JavaScript.")
                time.sleep(1)
                # --------------------------

                # --- Click the FIRST "Create Animation" button ---
                # This locator should target the button outside the dialog
                first_create_animation_button_locator = (By.XPATH, "//button[normalize-space(.)='Create Animation' and not(ancestor::div[contains(@class, 'MuiDialogActions-root')])]")
                print(f"Waiting for FIRST Create Animation button with locator: {first_create_animation_button_locator}")

                # Wait for the button to be present (use presence for JS click)
                first_create_animation_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(first_create_animation_button_locator)
                )
                print("FIRST Create Animation button found.")

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", first_create_animation_button)
                print("Clicked the FIRST Create Animation button using JavaScript.")
                time.sleep(3) # Increase pause slightly to allow dialog/state to update
                # --------------------------

                # --- Click the SECOND "Create Animation" button (within the dialog) ---
                # Locator specifically targets the button inside the MuiDialogActions div
                second_create_animation_button_locator = (By.XPATH, "//div[contains(@class, 'MuiDialogActions-root')]//button[normalize-space(.)='Create Animation']")
                print(f"Waiting for SECOND Create Animation button (in dialog) with locator: {second_create_animation_button_locator}")

                # Wait for the button to be present within the dialog
                second_create_animation_button = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(second_create_animation_button_locator)
                )
                print("SECOND Create Animation button (in dialog) found.")

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", second_create_animation_button)
                print("Clicked the SECOND Create Animation button (in dialog) using JavaScript.")
                time.sleep(2) # Pause after clicking second create
                # --------------------------
                initial_download_button_locator = (By.XPATH, "//button[.//span[normalize-space(text())='Download']]")
                print(f"Waiting for initial Download button with locator: {initial_download_button_locator}")
                initial_download_button = WebDriverWait(driver, 300).until( # Wait up to 5 minutes for processing
                    EC.presence_of_element_located(initial_download_button_locator)
                )
                print("Initial Download button found.")
                # Click the initial download button using JavaScript
                driver.execute_script("arguments[0].click();", initial_download_button)
                print("Clicked the initial Download button using JavaScript.")
                time.sleep(1) # Short pause after clicking initial download button
                # --------------------------

                # --- Wait for processing and initiate download ---
                # Wait for the button group containing 'bvh'/'mp4' to appear
                bvh_button_locator = (By.XPATH, "//div[contains(@class, 'MuiButtonGroup-root')]//button[normalize-space(.)='bvh']")
                print(f"Waiting for 'bvh' button (within group) with locator: {bvh_button_locator}")
                bvh_button = WebDriverWait(driver, 300).until( # Wait up to 5 minutes for processing
                    EC.presence_of_element_located(bvh_button_locator)
                )
                print("'bvh' button found.")
                # Click the 'bvh' button - assuming this opens the format menu
                driver.execute_script("arguments[0].click();", bvh_button)
                print("Clicked the 'bvh' button using JavaScript.")
                time.sleep(1) # Wait for dropdown menu to open
                # --------------------------

                # --- Skip clicking initial download button and dropdown arrow ---
                # # Click the initial download button using JavaScript
                # driver.execute_script("arguments[0].click();", initial_download_button)
                # print("Clicked the initial Download button using JavaScript.")
                # time.sleep(3) # Short pause after clicking initial download button
                #
                # # Click the dropdown arrow BUTTON next to the download button
                # # Locate the BUTTON containing the dropdown SVG
                # dropdown_button_locator = (By.XPATH, "//button[.//svg[@data-testid='ArrowDropDownIcon']]")
                # print(f"Waiting for dropdown button with locator: {dropdown_button_locator}")
                # dropdown_button = WebDriverWait(driver, 10).until(
                #     EC.presence_of_element_located(dropdown_button_locator) # Wait for the button
                # )
                # print("Dropdown button found.")
                # driver.execute_script("arguments[0].click();", dropdown_button) # Click the button
                # print("Clicked the dropdown button using JavaScript.")
                # time.sleep(1) # Wait for dropdown menu to open
                # --------------------------

                # Click the "mp4" menu item (should appear after clicking 'bvh' button)
                mp4_menu_item_locator = (By.XPATH, "//li[normalize-space(.)='mp4']")
                print(f"Waiting for 'mp4' menu item with locator: {mp4_menu_item_locator}")
                mp4_menu_item = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(mp4_menu_item_locator)
                )
                print("'mp4' menu item found.")
                driver.execute_script("arguments[0].click();", mp4_menu_item)
                print("Clicked 'mp4' menu item using JavaScript.")
                time.sleep(1) # Wait for selection to register

                # Click the final "Download" button
                # This locator should still be valid after selecting mp4
                final_download_button_locator = (By.XPATH, "//div[contains(@class, 'mt-3')]//button[normalize-space(.)='Download']")
                # Alternative using specific class:
                # final_download_button_locator = (By.CSS_SELECTOR, "div.mt-3 button.css-p007i2")
                print(f"Waiting for final Download button with locator: {final_download_button_locator}")
                final_download_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(final_download_button_locator)
                )
                print("Final Download button found.")
                driver.execute_script("arguments[0].click();", final_download_button)
                print("Clicked the final Download button using JavaScript.")
                time.sleep(5) # Pause to allow download to start
                # --------------------------


            except Exception as upload_error:
                # Updated error message
                print(f"Error during file upload, switch clicks, MP4 settings, Create Animation clicks, or Download: {upload_error}") # Updated message
                driver.save_screenshot("upload_error_screenshot.png")
                print("Saved screenshot to upload_error_screenshot.png")
        # ----------------------------------------------------

    except Exception as e:
        # Updated error message to include potential upload errors
        print(f"An error occurred during the process: {e}")
        # Add more detailed error info if possible
        try:
            # Try to get page source or screenshot for debugging
            print("Page Source at time of error:")
            # print(driver.page_source) # Uncomment cautiously - can be very long
            driver.save_screenshot("error_screenshot.png")
            print("Saved screenshot to error_screenshot.png")
        except Exception as e_debug:
            print(f"Could not get debug info: {e_debug}")


    # Keep the browser open for a while (e.g., 5 seconds) to see the result
    time.sleep(5)

    # Close the browser window
    driver.quit()

if __name__ == "__main__":
    # Example usage
    file_name = "hello"  # Replace with your actual file name (without extension)
    generate_avatar(file_name)
    print("Avatar generation process completed.")
