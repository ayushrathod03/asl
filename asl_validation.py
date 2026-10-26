# asl_realtime.py
# Real-time ASL recognition using MediaPipe + KNN + Autocomplete

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

import os

# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark_knn.joblib")
WORDLIST_PATH = os.path.join(BASE_DIR, "words.txt")
BUFFER_SIZE = 11              # sliding window for smoothing
CONFIRM_THRESHOLD = 0.8      # confidence threshold for accepting a prediction
SHOW_AUTOCOMPLETE = True     # set False if you just want letter output
# -------------------------

# Load model
print("Loading model...")
data = joblib.load(MODEL_PATH)
knn = data["knn"]
pca = data.get("pca", None)
labels = data["labels"]
print("Model loaded successfully.")

# Load word list
try:
    with open(WORDLIST_PATH, "r") as f:
        WORDS = [w.strip().lower() for w in f if w.strip()]
    print(f"Loaded {len(WORDS)} words from words.txt")
except FileNotFoundError:
    WORDS = ["hello", "help", "good", "morning", "sign", "language", "test", "thank", "you", "yes", "no"]
    print("words.txt not found, using small default list.")

# Autocomplete (Trie-like prefix search)
def suggest(prefix):
    if len(prefix) == 0:
        return []
    prefix = prefix.lower()
    return [w for w in WORDS if w.startswith(prefix)][:5]

# Hand detector setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam. Check camera permissions.")

buffer = deque(maxlen=BUFFER_SIZE)
typed = ""
print("Press 'q' to quit, 'c' to clear typed text.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed, exiting.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    predictions = []

    if res.multi_hand_landmarks:
        for idx, lm in enumerate(res.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
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

        buffer.append(predictions[0])
    else:
        buffer.append("Nothing")

    # Smooth prediction
    if len(buffer) == BUFFER_SIZE:
        most_common, count = Counter(buffer).most_common(1)[0]
        confidence = count / BUFFER_SIZE
        if most_common != "Nothing" and confidence > CONFIRM_THRESHOLD:
            typed += most_common
            buffer.clear()

            import time
            time.sleep(1)   # pause 1 second after registering a letter


    # Autocomplete suggestions
    suggestions = suggest(typed) if SHOW_AUTOCOMPLETE else []

    # UI display
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    pred_text = " | ".join([f"H{i+1}: {p}" for i, p in enumerate(predictions)]) if predictions else "Nothing"
    cv2.putText(frame, f"Live Pred: {pred_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Typed: {typed}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if suggestions:
        for i, w in enumerate(suggestions):
            cv2.putText(frame, w, (10, 85 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("ASL Real-Time Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        typed = ""
        buffer.clear()

cap.release()
cv2.destroyAllWindows()
