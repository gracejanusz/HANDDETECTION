import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
import joblib
import os
from io import BytesIO

# --- Configuration & Model Loading ---

class HandGestureModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

@st.cache_resource
def load_resources(model_path='hand_gesture_model.pth', encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 63
        num_classes = 24
        model = HandGestureModel(input_size, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        return model, label_encoder, scaler, hands, device
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None

# --- Streamlit App Setup ---

st.set_page_config(page_title="ASL Trainer | BridgeSign", page_icon="🤟", layout="wide")

model, label_encoder, scaler, hands, device = load_resources()

if not all([model, label_encoder, scaler, hands, device]):
    st.stop()

LETTERS = list("abcdefghiklmnopqrstuvwxy")

# --- Session State Initialization ---

if 'current_letter_index' not in st.session_state:
    st.session_state.current_letter_index = 0
if 'show_next' not in st.session_state:
    st.session_state.show_next = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_captured_image' not in st.session_state:
    st.session_state.last_captured_image = None
if 'capture_pressed' not in st.session_state:
    st.session_state.capture_pressed = False

# --- Styling ---

st.markdown("""
    <style>
    .stApp { background-color: #f5f5dc !important; color: black !important; }
    div[data-testid="stHorizontalBlock"] { height: 100%; }
    
    /* Left Panel (Reference Image Panel) */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]) {
        background-color:  #ffe9a5;
        padding: 20px;
        border-radius: 10px;
        height: 80vh;
    }

    /* Right Panel (Camera Input Panel) */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stCameraInput"]) {
        background-color: #0277b5;
        padding: 20px;
        border-radius: 10px;
        height: 80vh;
    }

    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain;
        border-radius: 5px;
    }

    .stButton>button { background-color:  #0277b5 !important; color: black !important; border-radius: 5px; font-size: 1.1em; margin-top: 10px; }
    .stButton>button:hover { background-color:  #2aaaff !important; }
    
    .feedback-correct { color: darkgreen; background-color: #e0ffe0; font-weight: bold; padding: 10px; border-radius: 5px; }
    .feedback-incorrect { color: darkred; background-color: #ffe0e0; font-weight: bold; padding: 10px; border-radius: 5px; }
    .feedback-info { color: #cc5500; background-color: #fff0e0; font-weight: bold; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_current_letter():
    idx = st.session_state.current_letter_index
    if 0 <= idx < len(LETTERS):
        return LETTERS[idx]
    return None

def load_asl_image(letter):
    path = os.path.join("asl_images", f"{letter}.png")
    if os.path.exists(path):
        return Image.open(path).resize((250, 250))
    return None

def classify_hand_from_frame(frame_data):
    if frame_data is None:
        return None
    image_bytes = frame_data.getvalue()
    pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
    frame = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return "No hand detected"
    landmarks_list = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks_list.extend([lm.x, lm.y, lm.z])
    landmarks_np = np.array(landmarks_list).reshape(1, -1)
    try:
        landmarks_scaled = scaler.transform(landmarks_np)
    except Exception:
        return "Scaling error"
    input_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        try:
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)
            predicted_index = prediction.item()
            predicted_letter = label_encoder.inverse_transform([predicted_index])[0]
            return predicted_letter.lower()
        except Exception:
            return "Prediction error"

# --- Main Layout ---

current_letter = get_current_letter()

if current_letter:
    col1, col2 = st.columns(2)

    with col2:
        captured_image_data = st.camera_input("Camera Feed", key=f"cam_{current_letter}", label_visibility="hidden")
        if captured_image_data:
            st.session_state.last_captured_image = captured_image_data

    with col1:
        st.subheader(f"Sign {current_letter.upper()} and take a photo inside the camera!")

        asl_img = load_asl_image(current_letter)
        if asl_img:
            st.image(asl_img, width=200)

        feedback_placeholder = st.empty()

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("Check!", key="check_btn", use_container_width=True):
                st.session_state.capture_pressed = True

        with btn_col2:
            next_disabled = not st.session_state.get('show_next', False)
            if st.button("Next Letter", key="next_btn", disabled=next_disabled, use_container_width=True):
                if st.session_state.last_captured_image is not None:
                    st.error("⚠️ Please clear the photo inside the camera before moving to the next letter!")
                else:
                    st.session_state.current_letter_index += 1
                    st.session_state.show_next = False
                    st.session_state.last_prediction = None
                    st.session_state.last_captured_image = None
                    st.rerun()

        # --- Classification and Feedback ---
        if st.session_state.capture_pressed:
            with st.spinner('Analyzing...'):
                last_image = st.session_state.get('last_captured_image', None)
                if last_image:
                    predicted_letter = classify_hand_from_frame(last_image)
                    st.session_state.last_prediction = predicted_letter

                    if predicted_letter == "No hand detected":
                        feedback_placeholder.markdown("<p class='feedback-info'>🖐️ No hand detected. Please try again.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                    elif predicted_letter in ["Scaling error", "Prediction error"]:
                        feedback_placeholder.markdown(f"<p class='feedback-incorrect'>⚙️ Error: {predicted_letter}.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                    elif predicted_letter == current_letter:
                        feedback_placeholder.markdown(f"<p class='feedback-correct'>✅ Correct! You signed '{predicted_letter.upper()}'! Now you can clear the photo and go to next letter!.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = True
                    else:
                        feedback_placeholder.markdown(f"<p class='feedback-incorrect'>❌ That looked like '{predicted_letter.upper()}'. Try signing '{current_letter.upper()}'.</p>", unsafe_allow_html=True)
                        st.session_state.show_next = False
                else:
                    feedback_placeholder.markdown("<p class='feedback-info'>📸 Please take a photo first.</p>", unsafe_allow_html=True)
                    st.session_state.show_next = False

            st.session_state.capture_pressed = False
            st.session_state.last_captured_image = None  # 🚨 Clear the photo after checking

else:
    st.balloons()
    st.success("🎉 Congratulations! You've completed all letters! 🎉")
    if st.button("Practice Again?"):
        st.session_state.current_letter_index = 0
        st.session_state.show_next = False
        st.session_state.last_prediction = None
        st.session_state.last_captured_image = None
        st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)
