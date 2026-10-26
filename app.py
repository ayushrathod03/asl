import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ASL Sign Predictor & Autocompleter",
    page_icon="🤟",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stApp {
        background: transparent;
    }
    .title-banner {
        text-align: center;
        padding: 20px;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .word-card {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .typed-box {
        font-size: 2rem;
        letter-spacing: 2px;
        font-weight: bold;
        color: #38bdf8;
        background: #0f172a;
        padding: 15px 25px;
        border-radius: 12px;
        border: 2px solid #38bdf8;
        text-align: center;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-banner">
    <h1>🤟 ASL Sign Predictor & Word Autocompleter</h1>
    <p style="color: #94a3b8; font-size: 1.1rem;">Real-time American Sign Language recognition powered by MediaPipe & KNN</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Resources
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark_knn.joblib")
WORDLIST_PATH = os.path.join(BASE_DIR, "words.txt")

@st.cache_resource
def load_resources():
    model_data = joblib.load(MODEL_PATH)
    knn = model_data["knn"]
    pca = model_data.get("pca", None)
    labels = model_data["labels"]
    
    try:
        with open(WORDLIST_PATH, "r") as f:
            words = [w.strip().lower() for w in f if w.strip()]
    except Exception:
        words = ["hello", "help", "good", "morning", "sign", "language", "test", "thank", "you", "yes", "no"]
        
    return knn, pca, labels, words

knn, pca, labels, WORDS = load_resources()

def suggest(prefix):
    if not prefix:
        return []
    prefix = prefix.lower()
    return [w for w in WORDS if w.startswith(prefix)][:5]

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "typed" not in st.session_state:
    st.session_state.typed = ""
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=11)

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
col_cam, col_info = st.columns([3, 2])

with col_info:
    st.subheader("📝 Typed Buffer")
    typed_placeholder = st.empty()
    typed_placeholder.markdown(f'<div class="typed-box">{st.session_state.typed if st.session_state.typed else "..."}</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Typed Text", use_container_width=True):
        st.session_state.typed = ""
        st.session_state.buffer.clear()
        st.rerun()

    st.subheader("💡 Autocomplete Suggestions")
    suggestions_placeholder = st.empty()
    suggestions = suggest(st.session_state.typed)
    if suggestions:
        suggestions_html = "".join([f'<div class="word-card">✨ {word}</div>' for word in suggestions])
        suggestions_placeholder.markdown(suggestions_html, unsafe_allow_html=True)
    else:
        suggestions_placeholder.info("Start signing letters to see word completions!")

with col_cam:
    st.subheader("📹 Live Camera Feed")
    run_cam = st.checkbox("Start Camera", value=True)
    FRAME_WINDOW = st.image([])

if run_cam:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check camera permissions.")
    else:
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab camera frame.")
                break
                
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            predictions = []
            
            if res.multi_hand_landmarks:
                for idx, lm in enumerate(res.multi_hand_landmarks):
                    mp_draw.draw_landmarks(rgb, lm, mp_hands.HAND_CONNECTIONS)
                    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                    pts[:, :2] -= pts[0, :2]
                    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
                    if scale > 1e-6:
                        pts[:, :2] /= scale
                    else:
                        pts[:, :2] = 0
                    vec = pts.flatten().reshape(1, -1)
                    if pca is not None:
                        vec = pca.transform(vec)
                    pred_label = knn.predict(vec)[0]
                    predictions.append(pred_label)
                    
                    cv2.putText(rgb, f"Hand {idx+1}: {pred_label}", (20, 40 + idx * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                st.session_state.buffer.append(predictions[0])
            else:
                st.session_state.buffer.append("Nothing")
                
            if len(st.session_state.buffer) == 11:
                most_common, count = Counter(st.session_state.buffer).most_common(1)[0]
                confidence = count / 11
                if most_common != "Nothing" and confidence > 0.8:
                    st.session_state.typed += most_common
                    st.session_state.buffer.clear()
                    typed_placeholder.markdown(f'<div class="typed-box">{st.session_state.typed if st.session_state.typed else "..."}</div>', unsafe_allow_html=True)
                    suggs = suggest(st.session_state.typed)
                    if suggs:
                        suggs_html = "".join([f'<div class="word-card">✨ {w}</div>' for w in suggs])
                        suggestions_placeholder.markdown(suggs_html, unsafe_allow_html=True)
                    else:
                        suggestions_placeholder.info("No matching words found.")
            
            FRAME_WINDOW.image(rgb)
        
        cap.release()
