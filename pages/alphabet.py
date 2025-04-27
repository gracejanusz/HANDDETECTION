import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
import joblib
import os

# --- Configuration & Model Loading ---

# Define the model structure (must match the trained model)
# Define the model structure (must match the *saved* model in .pth)
class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Layers based on the state_dict from hand_gesture_model.pth
        self.fc1 = torch.nn.Linear(input_size, 256) # Checkpoint expects [256, 63]
        self.fc2 = torch.nn.Linear(256, 128)       # Checkpoint expects [128, 256]
        self.fc3 = torch.nn.Linear(128, 64)        # Checkpoint expects [64, 128]
        self.fc4 = torch.nn.Linear(64, num_classes) # Checkpoint expects fc4 with output num_classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x)) # Add relu activation for fc3
        x = self.fc4(x)             # Output from fc4
        return x

# Function to load model and resources (cached)
@st.cache_resource # Cache resources for efficiency
def load_resources(model_path='hand_gesture_model.pth', encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    """Loads the model, label encoder, and scaler."""
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        input_size = 63  # 21 landmarks * 3 coordinates (x, y, z)
        num_classes = 24 # a-z excluding j, z
        model = HandGestureModel(input_size, num_classes).to(device)
        # Load state dict, ensuring compatibility with the device
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode

        # Load label encoder and scaler
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)

        # Load MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

        return model, label_encoder, scaler, hands, device
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}. Make sure model, encoder, and scaler files are in the root directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None, None, None, None, None

# --- Streamlit App ---

st.set_page_config(page_title="ASL Trainer | BridgeSign", page_icon="🤟", layout="wide")

# Load resources
model, label_encoder, scaler, hands, device = load_resources()

# Check if resources loaded successfully
if not all([model, label_encoder, scaler, hands, device]):
    st.warning("Could not load necessary resources. Please check file paths and try again.")
    st.stop() # Stop execution if resources aren't loaded

# Define the letters to practice
LETTERS = list("abcdefghiklmnopqrstuvwxy") # Skipping j, z

# Initialize session state
if 'current_letter_index' not in st.session_state:
    st.session_state.current_letter_index = 0
if 'show_next' not in st.session_state:
    st.session_state.show_next = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
    if 'last_captured_image' not in st.session_state:
        st.session_state.last_captured_image = None

# --- Styling for Mockup ---
st.markdown(
    """
    <style>
    /* Overall App Background */
    .stApp {
        background-color: #f5f5dc !important; /* Beige */
        color: black !important; /* Default text color */
    }

    /* Remove padding around the main block container */
     .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }

    /* Ensure columns take full height potentially */
    div[data-testid="stHorizontalBlock"] {
        height: 100%; /* Try to make columns fill height */
    }

    /* Left Panel Styling (Blue) */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]) { /* Target left column */
        background-color: #add8e6; /* Light Blue */
        padding: 20px;
        border-radius: 10px;
        height: 80vh; /* Make panel take significant height */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Pushes buttons to bottom */
    }

    /* Right Panel Styling (Green) */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stCameraInput"]) { /* Target right column */
        background-color: #90ee90; /* Light Green */
        padding: 20px;
        border-radius: 10px;
        height: 80vh; /* Make panel take significant height */
        display: flex; /* Allow camera to fill */
        flex-direction: column;
    }

    /* Camera Input Styling */
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        height: 100% !important; /* Make container fill panel */
        display: flex;
        flex-direction: column;
        flex-grow: 1; /* Allow it to take available space */
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: 100% !important; /* Make video fill container */
        object-fit: contain; /* Fit video within bounds */
        border-radius: 5px;
    }
    /* Hide the default camera button - we use our own */
     div[data-testid="stCameraInput"] button {
         display: none !important;
     }

    /* Instruction Text */
    h3 { /* Target st.subheader */
        color: black !important;
        text-align: left;
        font-size: 1.5em;
        margin-bottom: 15px;
    }

    /* Reference Image Container */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center; /* Center image horizontally */
        margin-bottom: 20px;
    }
    div[data-testid="stImage"] img {
         max-width: 100%; /* Ensure image fits */
         max-height: 40vh; /* Limit image height */
         height: auto;
         border-radius: 5px;
    }


    /* Button Styling (Capture & Next) */
    .stButton>button {
        background-color: #90ee90 !important; /* Light Green */
        color: black !important;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1em;
        width: 100% !important; /* Make buttons fill their columns */
        margin-top: 10px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #7fdd7f !important; /* Slightly darker green */
        color: black !important;
    }
    .stButton>button:disabled {
        background-color: #cccccc !important;
        color: #666666 !important;
        cursor: not-allowed;
    }

    /* Feedback Message Styling */
    .feedback-correct, .feedback-incorrect, .feedback-info {
        font-size: 1.1em;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
        padding: 10px;
        border-radius: 5px;
    }
    .feedback-correct { color: darkgreen; background-color: #e0ffe0; }
    .feedback-incorrect { color: darkred; background-color: #ffe0e0; }
    .feedback-info { color: #cc5500; background-color: #fff0e0; } /* Orange-ish */

    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---

def get_current_letter():
    """Gets the current letter based on the session state index."""
    idx = st.session_state.current_letter_index
    if 0 <= idx < len(LETTERS):
        return LETTERS[idx]
    return None

def load_asl_image(letter):
    """Loads and returns the ASL image for the given letter."""
    path = os.path.join("asl_images", f"{letter}.png") # Reverted path
    if os.path.exists(path):
        try:
            img = Image.open(path).resize((250, 250)) # Resize for display
            return img
        except Exception as e:
            st.warning(f"Failed to load image for '{letter}': {e}")
            return None
    else:
        st.warning(f"Missing ASL image for '{letter.upper()}' at path: {path}")
        return None

def classify_hand_from_frame(frame_data):
    """Classifies the hand gesture from a captured frame."""
    if frame_data is None:
        return None

    # Convert PIL Image/UploadedFile to OpenCV format (BGR)
    pil_image = Image.open(frame_data).convert('RGB')
    frame = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # MediaPipe needs RGB
    result = hands.process(frame_rgb)

    if not result.multi_hand_landmarks:
        return "No hand detected" # Specific feedback

    landmarks_list = []
    # Assuming only one hand is detected as per Hands configuration
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks_list.extend([lm.x, lm.y, lm.z])

    # Prepare data for the model
    landmarks_np = np.array(landmarks_list).reshape(1, -1)
    try:
        landmarks_scaled = scaler.transform(landmarks_np) # Normalize using saved scaler
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return "Scaling error"

    input_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        try:
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)
            predicted_index = prediction.item()
            predicted_letter = label_encoder.inverse_transform([predicted_index])[0]
            return predicted_letter.lower() # Return lowercase for comparison
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return "Prediction error"


# --- App Layout ---

current_letter = get_current_letter()

if current_letter:
    # --- Layout Setup ---
    col1, col2 = st.columns(2) # 50/50 split

    # --- Right Panel (Camera) ---
    with col2:
        # Container for green background and camera
        with st.container(): # Add class="right-panel" via CSS later
            # Camera Input - store image in session state when taken
            captured_image_data = st.camera_input(
                "Camera Feed",
                key=f"cam_{current_letter}",
                label_visibility="hidden" # Hide the label "Camera Feed"
            )
            if captured_image_data:
                st.session_state.last_captured_image = captured_image_data

    # --- Left Panel (Instructions, Image, Buttons, Feedback) ---
    with col1:
        # Container for blue background
        with st.container(): # Add class="left-panel" via CSS later
            # Instruction
            st.subheader(f"Sign {current_letter.upper()}!") # Simpler instruction

            # Reference Image
            asl_img = load_asl_image(current_letter)
            if asl_img:
                # Center the image - Use columns to help center
                img_col, _, _ = st.columns([1, 1, 1])
                with img_col:
                     st.image(asl_img, width=200) # Fixed width for consistency
            else:
                st.error("Could not load reference image.")

            # Placeholder for feedback messages
            feedback_placeholder = st.empty()

            # Buttons Row
            btn_col1, btn_col2 = st.columns(2)
            capture_pressed = False
            with btn_col1:
                 # Add class="capture-button" via CSS later
                if st.button("Capture", key="capture_btn", use_container_width=True):
                    capture_pressed = True

            with btn_col2:
                 # Add class="next-button" via CSS later
                next_disabled = not st.session_state.get('show_next', False)
                if st.button("Next", key="next_btn", disabled=next_disabled, use_container_width=True):
                    st.session_state.current_letter_index += 1
                    st.session_state.show_next = False
                    st.session_state.last_prediction = None
                    st.session_state.last_captured_image = None # Clear captured image too
                    st.rerun() # Rerun to load next letter

            # --- Classification Logic (Triggered by Capture Button) ---
            if capture_pressed:
                last_image = st.session_state.get('last_captured_image', None)
                if last_image:
                    predicted_letter = classify_hand_from_frame(last_image)
                    st.session_state.last_prediction = predicted_letter # Store prediction

                    # Display Feedback
                    if predicted_letter == "No hand detected":
                        feedback_placeholder.markdown("<p class='feedback-info'>🖐️ No hand detected. Please try again.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                    elif predicted_letter in ["Scaling error", "Prediction error"]:
                        feedback_placeholder.markdown(f"<p class='feedback-incorrect'>⚙️ Error: {predicted_letter}.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                    elif predicted_letter == current_letter:
                        feedback_placeholder.markdown(f"<p class='feedback-correct'>✅ Correct! You signed '{predicted_letter.upper()}'.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = True # Enable the 'Next' button
                    else:
                        feedback_placeholder.markdown(f"<p class='feedback-incorrect'>❌ That looked like '{predicted_letter.upper()}'. Try signing '{current_letter.upper()}'.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                else:
                    # No image captured yet
                    feedback_placeholder.markdown("<p class='feedback-info'>📸 Please capture an image first.</p>", unsafe_allow_html=True)
                    st.session_state.show_next = False

            # Persist feedback if already shown (and no new capture pressed)
            elif st.session_state.last_prediction is not None and not capture_pressed:
                 predicted_letter = st.session_state.last_prediction
                 if predicted_letter == "No hand detected":
                     feedback_placeholder.markdown("<p class='feedback-info'>🖐️ No hand detected. Please try again.</p>", unsafe_allow_html=True)
                 elif predicted_letter in ["Scaling error", "Prediction error"]:
                     feedback_placeholder.markdown(f"<p class='feedback-incorrect'>⚙️ Error: {predicted_letter}.</p>", unsafe_allow_html=True)
                 elif predicted_letter == current_letter:
                     feedback_placeholder.markdown(f"<p class='feedback-correct'>✅ Correct! You signed '{predicted_letter.upper()}'.</p>", unsafe_allow_html=True)
                     # Ensure next is enabled if correct prediction persists
                     st.session_state.show_next = True
                 else:
                     feedback_placeholder.markdown(f"<p class='feedback-incorrect'>❌ That looked like '{predicted_letter.upper()}'. Try signing '{current_letter.upper()}'.</p>", unsafe_allow_html=True)


else:
    # --- End of Practice Session ---
    st.balloons()
    st.success("🎉 Congratulations! You've completed all the letters! 🎉")
    if st.button("Practice Again?"):
        st.session_state.current_letter_index = 0
        st.session_state.show_next = False
        st.session_state.last_prediction = None
        st.rerun()

# ---- Footer ----
st.markdown("<br><br>", unsafe_allow_html=True) 