import streamlit as st
import numpy as np
import os
import sys
import time

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model.h5")
IMG_SIZE = (48, 48)

# ─────────────────────────────────────────────
# LOAD TENSORFLOW / KERAS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TensorFlow…")
def load_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model  # noqa: F401
        print("[INFO] TensorFlow loaded:", tf.__version__)
        return tf
    except ImportError as e:
        print(f"[ERROR] TensorFlow import failed: {e}")
        return None

tf = load_tf()

# ─────────────────────────────────────────────
# LOAD EMOTION MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading emotion model…")
def load_emotion_model(path: str):
    if tf is None:
        print("[ERROR] TensorFlow not available — cannot load model.")
        return None, "TensorFlow is not installed or failed to import."

    if not os.path.isfile(path):
        msg = f"Model file not found at: {path}"
        print(f"[ERROR] {msg}")
        return None, msg

    try:
        from tensorflow.keras.models import load_model
        model = load_model(path, compile=False)
        print(f"[INFO] Model loaded successfully from: {path}")
        print(f"[INFO] Model input shape: {model.input_shape}")
        return model, None
    except Exception as e:
        msg = f"Failed to load model: {e}"
        print(f"[ERROR] {msg}")
        return None, msg

# ─────────────────────────────────────────────
# LOAD OPENCV + HAAR CASCADE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OpenCV…")
def load_opencv():
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            msg = f"Haar cascade failed to load from: {cascade_path}"
            print(f"[ERROR] {msg}")
            return None, None, msg
        print(f"[INFO] OpenCV version: {cv2.__version__}")
        print(f"[INFO] Haar cascade loaded from: {cascade_path}")
        return cv2, face_cascade, None
    except ImportError as e:
        msg = f"OpenCV import failed: {e}"
        print(f"[ERROR] {msg}")
        return None, None, msg

# ─────────────────────────────────────────────
# OPEN WEBCAM
# ─────────────────────────────────────────────
def open_webcam(cv2, camera_index: int = 0):
    """Try DirectShow first (Windows), then fallback to default."""
    backends = []
    if sys.platform == "win32":
        try:
            backends.append(("DirectShow", cv2.CAP_DSHOW))
        except AttributeError:
            pass
    backends.append(("Default", cv2.CAP_ANY))

    for name, backend in backends:
        print(f"[INFO] Trying webcam {camera_index} with backend: {name}")
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Webcam initialized successfully with backend: {name}")
                return cap, None
            cap.release()
            print(f"[WARN] Webcam opened but first frame read failed with backend: {name}")
        else:
            print(f"[WARN] Webcam failed to open with backend: {name}")

    msg = (
        f"Could not open webcam at index {camera_index}. "
        "Check that your webcam is connected and not in use by another application."
    )
    print(f"[ERROR] {msg}")
    return None, msg

# ─────────────────────────────────────────────
# PREDICT EMOTION FROM FACE ROI
# ─────────────────────────────────────────────
def predict_emotion(model, cv2, face_roi_gray):
    """
    face_roi_gray: grayscale numpy array (any size) cropped to the face region.
    Returns (label, confidence_dict).
    """
    try:
        resized = cv2.resize(face_roi_gray, IMG_SIZE)
        normalized = resized.astype("float32") / 255.0
        input_tensor = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
        preds = model.predict(input_tensor, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        label = EMOTION_LABELS[top_idx]
        confidences = {EMOTION_LABELS[i]: float(preds[i]) for i in range(len(EMOTION_LABELS))}
        print(f"[INFO] Prediction → {label} ({preds[top_idx]*100:.1f}%)")
        return label, confidences
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise

# ─────────────────────────────────────────────
# DRAW DETECTIONS ON FRAME
# ─────────────────────────────────────────────
def draw_detections(cv2, frame, faces, labels):
    COLORS = {
        "Happy": (0, 255, 100),
        "Sad": (255, 100, 0),
        "Angry": (0, 0, 255),
        "Surprise": (0, 200, 255),
        "Fear": (128, 0, 128),
        "Disgust": (0, 128, 0),
        "Neutral": (200, 200, 200),
    }
    for (x, y, w, h), label in zip(faces, labels):
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
        )
    return frame

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🎭 Face Emotion Detection")
st.caption("Real-time emotion recognition via webcam · OpenCV + TensorFlow/Keras")

# ── Load resources ──────────────────────────
model, model_err = load_emotion_model(MODEL_PATH)
cv2, face_cascade, cv_err = load_opencv()

# ── Show load errors prominently ────────────
if model_err:
    st.error(f"❌ **Model Error:** {model_err}")
    st.info(
        f"Expected model path: `{MODEL_PATH}`\n\n"
        "Make sure `emotion_model.h5` is in the **same folder** as `app.py` and re-run."
    )
    st.stop()

if cv_err:
    st.error(f"❌ **OpenCV Error:** {cv_err}")
    st.info("Install OpenCV: `pip install opencv-python`")
    st.stop()

st.success("✅ Model and OpenCV loaded successfully.")

# ── Sidebar controls ─────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    scale_factor = st.slider("Detection scale factor", 1.05, 1.5, 1.3, 0.05)
    min_neighbors = st.slider("Min neighbors", 1, 10, 5, 1)
    confidence_threshold = st.slider("Min confidence to display (%)", 0, 100, 0, 5)
    show_bars = st.checkbox("Show confidence bars", value=True)
    st.divider()
    st.markdown("**Debug info**")
    st.code(f"Model: {MODEL_PATH}\nInput shape: {model.input_shape}", language="")

# ── Main controls ────────────────────────────
col1, col2 = st.columns(2)
start_btn = col1.button("▶ Start Webcam", use_container_width=True, type="primary")
stop_btn  = col2.button("⏹ Stop",         use_container_width=True)

if stop_btn:
    st.session_state["running"] = False

if start_btn:
    st.session_state["running"] = True

# ── Webcam loop ──────────────────────────────
if st.session_state.get("running", False):
    cap, cam_err = open_webcam(cv2, int(camera_index))

    if cam_err:
        st.error(f"❌ **Webcam Error:** {cam_err}")
        st.session_state["running"] = False
        st.stop()

    frame_placeholder = st.empty()
    emotion_placeholder = st.empty()
    status_placeholder  = st.empty()

    status_placeholder.info("🟢 Webcam running — press **Stop** to end.")

    frame_count = 0
    try:
        while st.session_state.get("running", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame — retrying…")
                time.sleep(0.05)
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=float(scale_factor),
                minNeighbors=int(min_neighbors),
                minSize=(48, 48),
            )

            if len(faces) == 0:
                print(f"[INFO] Frame {frame_count}: No face detected")
                label_list = []
            else:
                print(f"[INFO] Frame {frame_count}: {len(faces)} face(s) detected")
                label_list = []
                conf_list  = []
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    label, confs = predict_emotion(model, cv2, roi_gray)
                    label_list.append(label)
                    conf_list.append(confs)

                frame = draw_detections(cv2, frame, faces, label_list)

                # Confidence bars
                if show_bars and conf_list:
                    with emotion_placeholder.container():
                        for idx, (lbl, confs) in enumerate(zip(label_list, conf_list)):
                            st.markdown(f"**Face {idx + 1}: {lbl}**")
                            for emotion, prob in sorted(confs.items(), key=lambda x: -x[1]):
                                pct = prob * 100
                                if pct >= confidence_threshold:
                                    st.progress(prob, text=f"{emotion}: {pct:.1f}%")

            # Convert BGR → RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Runtime error: {e}")
        print(f"[ERROR] Runtime exception: {e}")
    finally:
        cap.release()
        print("[INFO] Webcam released.")
        status_placeholder.warning("⏹ Webcam stopped.")
        st.session_state["running"] = False

else:
    st.info("👆 Press **Start Webcam** to begin real-time emotion detection.")

