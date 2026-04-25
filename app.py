import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
# ==================== CONFIGURATION & SETUP ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model.h5"
st.set_page_config(
    page_title="Face Mood Detection",
    page_icon="ðŸ˜Š",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ==================== CUSTOM CSS FOR FUTURISTIC UI ====================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    body {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0f0f2e 100%);
        color: #e0e0ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        overflow-x: hidden;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a4e 50%, #0f0f2e 100%);
    }
    [data-testid="stMainBlockContainer"] {
        background: transparent;
        padding-top: 2rem;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        animation: glow-pulse 2s ease-in-out infinite;
    }
    .subtitle {
        text-align: center;
        color: #a0a0ff;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
        animation: fade-in-up 0.8s ease-out;
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 212, 255, 0.2);
    }
    .emotion-label {
        text-align: center;
        font-size: 4rem;
        margin: 1rem 0;
        animation: bounce-in 0.6s ease-out;
    }
    .emotion-name {
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    .confidence-container {
        margin: 1.5rem 0;
    }
    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #c0c0ff;
    }
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    .confidence-bar {
        height: 100%;
        background: linear-gradient(90deg, #00d4ff 0%, #7f5af0 100%);
        border-radius: 10px;
        transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .ai-response {
        background: linear-gradient(135deg, rgba(127, 90, 240, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        border-left: 4px solid #00d4ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        font-style: italic;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #d0d0ff;
        animation: fade-in 0.8s ease-out 0.4s both;
    }
    .button-primary {
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4);
    }
    .webcam-container {
        position: relative;
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid rgba(0, 212, 255, 0.4);
        background: rgba(0, 0, 0, 0.5);
    }
    .webcam-container img {
        width: 100%;
        display: block;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    }
    .status-online {
        background-color: #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }
    .status-offline {
        background-color: #ff4444;
        box-shadow: 0 0 10px #ff4444;
    }
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        flex: 1;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #a0a0ff;
        margin-top: 0.5rem;
    }
    /* Animations */
    @keyframes glow-pulse {
        0%, 100% { text-shadow: 0 0 10px rgba(0, 212, 255, 0.5); }
        50% { text-shadow: 0 0 20px rgba(127, 90, 240, 0.8); }
    }
    @keyframes fade-in-up {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes bounce-in {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4) !important;
    }
    h1, h2, h3 {
        color: #e0e0ff !important;
    }
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #00d4ff 0%, #7f5af0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stWarning {
        background: rgba(255, 150, 0, 0.15) !important;
        border: 1px solid rgba(255, 150, 0, 0.3) !important;
        border-radius: 10px !important;
    }
    .stError {
        background: rgba(255, 68, 68, 0.15) !important;
        border: 1px solid rgba(255, 68, 68, 0.3) !important;
        border-radius: 10px !important;
    }
    .stSuccess {
        background: rgba(0, 255, 0, 0.15) !important;
        border: 1px solid rgba(0, 255, 0, 0.3) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)
# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def load_emotion_model():
    """Load emotion detection model with proper error handling"""
    try:
        from tensorflow.keras.models import load_model
        if not MODEL_PATH.exists():
            st.error(f"âŒ Model file not found at: {MODEL_PATH}")
            st.info("Please ensure 'emotion_model.h5' is in the same directory as app.py")
            return None
        print(f"[DEBUG] Loading model from: {MODEL_PATH}")
        model = load_model(str(MODEL_PATH))
        print("[DEBUG] âœ… Model loaded successfully!")
        return model
    except ImportError:
        st.error("âŒ TensorFlow not installed. Run: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"âŒ Error loading model: {str(e)}")
        print(f"[DEBUG] Model loading error: {e}")
        return None

@st.cache_resource
def load_face_cascade():
    """Load Haar Cascade face detector"""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            st.error("âŒ Failed to load Haar Cascade classifier")
            return None
        print("[DEBUG] âœ… Haar Cascade loaded successfully!")
        return cascade
    except Exception as e:
        st.error(f"âŒ Error loading Haar Cascade: {str(e)}")
        return None

def detect_faces(frame, cascade):
    """Detect faces in frame"""
    if cascade is None or frame is None:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        return faces
    except Exception as e:
        print(f"[DEBUG] Face detection error: {e}")
        return []

def preprocess_face(face_img):
    """Preprocess face for emotion model"""
    try:
        if face_img is None or face_img.size == 0:
            return None
        # Resize to 48x48 and convert to grayscale
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized.astype("float32") / 255.0
        face_reshaped = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=-1)
        return face_reshaped
    except Exception as e:
        print(f"[DEBUG] Face preprocessing error: {e}")
        return None

def predict_emotion(model, face_img):
    """Predict emotion from face"""
    if model is None:
        return None, None
    try:
        processed = preprocess_face(face_img)
        if processed is None:
            return None, None
        predictions = model.predict(processed, verbose=0)
        emotion_idx = int(np.argmax(predictions[0]))
        return emotion_idx, predictions[0]
    except Exception as e:
        print(f"[DEBUG] Emotion prediction error: {e}")
        return None, None

def get_emotion_data(emotion_idx):
    """Get emotion label, emoji, and AI response"""
    emotions = {
        0: {"label": "Angry", "emoji": "ðŸ˜ ", "color": "#ff4444"},
        1: {"label": "Disgust", "emoji": "ðŸ¤¢", "color": "#ff8844"},
        2: {"label": "Fear", "emoji": "ðŸ˜¨", "color": "#9944ff"},
        3: {"label": "Happy", "emoji": "ðŸ˜„", "color": "#44ff44"},
        4: {"label": "Neutral", "emoji": "ðŸ˜", "color": "#88ccff"},
        5: {"label": "Sad", "emoji": "ðŸ˜¢", "color": "#4488ff"},
        6: {"label": "Surprise", "emoji": "ðŸ˜®", "color": "#ffcc44"}
    }
    ai_responses = {
        0: "I sense some frustration or anger. Take a deep breath. What's bothering you? I'm here to listen.",
        1: "It looks like something's turned you off. That's okay-we all have moments like these.",
        2: "I notice fear or anxiety. Remember, you're stronger than you think. What can I help with?",
        3: "Beautiful! Your happiness is contagious. Keep shining! ðŸŒŸ",
        4: "You seem calm and collected. A peaceful mind is a powerful mind.",
        5: "I sense sadness. It's okay to feel down sometimes. Would you like to talk about it?",
        6: "Wow! You look pleasantly surprised! What's the good news? ðŸŽ‰"
    }
    return emotions.get(emotion_idx, emotions[4]), ai_responses.get(emotion_idx, "")

def capture_webcam_frame():
    """Capture a single frame from the default webcam with Windows-friendly fallbacks."""
    backend_attempts = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    for backend in backend_attempts:
        cap = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Give the camera a brief warmup so first frame is usable on Windows laptops.
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print("[DEBUG] âœ… Webcam frame captured successfully!")
            return frame
    print("[DEBUG] Webcam failed to initialize")
    return None

def load_image_from_bytes(uploaded_file):
    """Decode an uploaded image or browser camera image into a BGR array."""
    try:
        raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if image is None:
            print("[DEBUG] Image decode failed")
        return image
    except Exception as e:
        print(f"[DEBUG] Image loading error: {e}")
        return None

def process_frame(frame):
    """Detect face, predict emotion, and return a display-ready RGB frame."""
    faces = detect_faces(frame, st.session_state.cascade)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    best_face = None
    best_area = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 212, 255), 3)
        area = w * h
        if area > best_area:
            best_area = area
            best_face = (x, y, w, h)
    if best_face is not None:
        x, y, w, h = best_face
        face_roi = frame[y:y + h, x:x + w]
        emotion_idx, confidences = predict_emotion(st.session_state.model, face_roi)
        if emotion_idx is not None and confidences is not None:
            st.session_state.last_emotion = emotion_idx
            st.session_state.last_confidence = confidences
            st.session_state.frame_count += 1
    return frame_rgb

# ==================== INITIALIZE SESSION STATE ====================
if "model" not in st.session_state:
    st.session_state.model = load_emotion_model()
if "cascade" not in st.session_state:
    st.session_state.cascade = load_face_cascade()
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "last_frame_rgb" not in st.session_state:
    st.session_state.last_frame_rgb = None
if "webcam_error" not in st.session_state:
    st.session_state.webcam_error = None
# ==================== MAIN UI ====================
st.markdown('<h1 class="main-title">ðŸŽ­ FACE MOOD DETECTION</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Emotion Recognition with AI</p>', unsafe_allow_html=True)
# Check if model loaded successfully
model_status = "ðŸŸ¢ Online" if st.session_state.model else "ðŸ”´ Offline"
cascade_status = "ðŸŸ¢ Online" if st.session_state.cascade else "ðŸ”´ Offline"
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Model Status:** {model_status}")
with col2:
    st.info(f"**Face Detector:** {cascade_status}")
# Create three-column layout
col_left, col_middle, col_right = st.columns([1, 1, 1], gap="large")
# ==================== LEFT COLUMN: INPUT SOURCE ====================
with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ðŸ“· Input Source")

    source = st.radio(
        "Choose how to provide a face image:",
        ["Browser Camera", "Upload Image", "Local Webcam (local only)"],
        index=0,
    )

    frame_placeholder = st.empty()
    frame = None
    st.session_state.webcam_error = None

    if source == "Browser Camera":
        camera_image = st.camera_input("Take a selfie")
        if camera_image is not None:
            frame = load_image_from_bytes(camera_image)
            if frame is None:
                st.session_state.webcam_error = "âŒ Unable to decode camera image."
            else:
                frame = cv2.flip(frame, 1)
    elif source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a face photo",
            type=["jpg", "jpeg", "png"],
            key="upload_image",
        )
        if uploaded_file is not None:
            frame = load_image_from_bytes(uploaded_file)
            if frame is None:
                st.session_state.webcam_error = "âŒ Unable to decode uploaded image."
    else:
        capture_btn = st.button("ðŸ“¸ Capture Frame", key="capture", use_container_width=True)
        if capture_btn:
            frame = capture_webcam_frame()
            if frame is None:
                st.session_state.webcam_error = "âŒ Local webcam not available."
            else:
                frame = cv2.flip(frame, 1)

    if frame is not None:
        st.session_state.last_frame_rgb = process_frame(frame)

    if st.session_state.last_frame_rgb is not None:
        frame_placeholder.image(st.session_state.last_frame_rgb, use_container_width=True)
    elif st.session_state.webcam_error:
        st.error(st.session_state.webcam_error)
    else:
        st.warning("ðŸ‘ï¸ Provide an image from the browser camera or upload a file to detect emotion.")

    if source == "Local Webcam (local only)":
        st.info("Local webcam capture only works when you run this app on your machine.")
    st.markdown('</div>', unsafe_allow_html=True)
# ==================== MIDDLE COLUMN: EMOTION PREDICTION ====================
with col_middle:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ðŸŽ¯ Emotion Analysis")
    if st.session_state.last_emotion is not None and st.session_state.last_confidence is not None:
        emotion_data, ai_response = get_emotion_data(st.session_state.last_emotion)
        confidence_scores = st.session_state.last_confidence
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        # Display emotion
        st.markdown(
            f'<div class="emotion-label">{emotion_data["emoji"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="emotion-name">{emotion_data["label"]}</div>',
            unsafe_allow_html=True
        )
        # Confidence
        top_confidence = float(confidence_scores[st.session_state.last_emotion])
        st.metric("Confidence", f"{top_confidence * 100:.1f}%", delta=None)
        # Confidence bars for all emotions
        st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
        for emotion, score in zip(emotions, confidence_scores):
            score_value = float(score)
            st.markdown(
                f'<div class="confidence-label"><span>{emotion}</span><span>{score_value * 100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="confidence-bar-bg"><div class="confidence-bar" '
                f'style="width: {score_value * 100:.1f}%"></div></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("ðŸ‘ï¸ No face detected yet. Enable live stream to start detection.")
    st.markdown('</div>', unsafe_allow_html=True)
# ==================== RIGHT COLUMN: AI INTERPRETATION ====================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ðŸ’­ AI Insight")
    if st.session_state.last_emotion is not None:
        emotion_data, ai_response = get_emotion_data(st.session_state.last_emotion)
        st.markdown(
            f'<div class="ai-response">{ai_response}</div>',
            unsafe_allow_html=True
        )
        # Stats
        st.markdown('<div class="stats-row">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{st.session_state.frame_count}</div>'
            f'<div class="stat-label">Frames Processed</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">AI</div>'
            f'<div class="stat-label">Powered</div></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ðŸ‘‹ Detected emotions will appear here with personalized AI insights!")
    st.markdown('</div>', unsafe_allow_html=True)
# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #a0a0ff; font-size: 0.9rem; margin-top: 2rem;">
    <p>🔬 Powered by TensorFlow • OpenCV • Streamlit</p>
    <p>Built for emotion recognition • Not for diagnosis</p>
    </div>
    """,
    unsafe_allow_html=True
)
print("[DEBUG] App initialized and running successfully!")
