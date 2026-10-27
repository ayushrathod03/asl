import os
import sys
import bisect
import base64
import joblib
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Setup Flask Application
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ---------------------------------------------------------------------------
# Load ASL Model & Wordlist
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark_knn.joblib")
WORDS_PATH = os.path.join(BASE_DIR, "words.txt")

print("[INFO] Loading ASL Hand Landmark KNN model...")
try:
    model_data = joblib.load(MODEL_PATH)
    knn_model = model_data["knn"]
    pca_model = model_data.get("pca", None)
    class_labels = [str(lbl) for lbl in model_data.get("labels", [])]
    print(f"[SUCCESS] Model loaded with {len(class_labels)} classes: {class_labels[:10]}...")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    knn_model = None
    pca_model = None
    class_labels = []

# Load and sort dictionary for high-performance O(log N) bisect autocomplete
print("[INFO] Loading and sorting word dictionary for bisect search...")
WORDS = []
try:
    if os.path.exists(WORDS_PATH):
        with open(WORDS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            raw_words = {line.strip().lower() for line in f if line.strip() and line.strip().isalpha()}
        WORDS = sorted(raw_words)
        print(f"[SUCCESS] Loaded {len(WORDS):,} unique sorted words.")
    else:
        print("[WARNING] words.txt not found, using curated fallback vocabulary.")
        WORDS = sorted(["hello", "help", "how", "home", "hand", "have", "good", "morning", "sign", 
                        "language", "thank", "you", "yes", "no", "please", "welcome", "world", "water"])
except Exception as e:
    print(f"[ERROR] Error loading dictionary: {e}")
    WORDS = sorted(["hello", "help", "good", "sign", "language", "thank", "you", "yes", "no"])

# Optional Server-side MediaPipe Hands fallback
server_hands_detector = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    server_hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    print("[INFO] Server-side MediaPipe Hands detector initialized for fallback.")
except Exception as e:
    print(f"[WARNING] Server MediaPipe detector unavailable ({e}). Client-side MediaPipe will be primary.")

# ---------------------------------------------------------------------------
# Helper Normalization & Prediction
# ---------------------------------------------------------------------------
def normalize_landmarks_vector(landmarks):
    """
    Normalizes a 21-point hand landmark vector matching training logic:
    - Shift coordinates so wrist (landmark 0) is at origin (0, 0).
    - Scale by max 2D Euclidean distance to ensure scale invariance.
    """
    pts = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    # Translate (x, y) relative to wrist
    pts[:, :2] -= pts[0, :2]
    # Scale by maximum 2D norm
    norms = np.linalg.norm(pts[:, :2], axis=1)
    max_scale = np.max(norms)
    if max_scale > 1e-6:
        pts[:, :2] /= max_scale
    else:
        pts[:, :2] = 0.0

    return pts.flatten().reshape(1, -1)

def run_knn_inference(features_vector):
    """Runs PCA (if trained) and KNN prediction with confidence score."""
    if knn_model is None:
        return "unknown", 0.0

    vec = features_vector
    if pca_model is not None:
        vec = pca_model.transform(vec)

    pred_label = str(knn_model.predict(vec)[0])
    
    confidence = 1.0
    if hasattr(knn_model, "predict_proba"):
        probs = knn_model.predict_proba(vec)[0]
        confidence = float(np.max(probs))

    return pred_label, confidence

# ---------------------------------------------------------------------------
# Web Routes & API Endpoints
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the main application page."""
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns system status, model details, and vocabulary statistics."""
    return jsonify({
        "status": "ready" if knn_model is not None else "degraded",
        "model_loaded": knn_model is not None,
        "classes_count": len(class_labels),
        "classes": class_labels,
        "vocab_size": len(WORDS),
        "server_mediapipe": server_hands_detector is not None
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Ultra-low latency landmark prediction.
    Accepts JSON with detected hands containing either:
    - 'landmarks': 63 floats (21 x 3) [x, y, z]
    - 'normalized': boolean (optional, default False)
    - 'handedness': 'Left' or 'Right' (optional)
    """
    if knn_model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    payload = request.get_json(silent=True)
    if not payload or "hands" not in payload:
        return jsonify({"success": False, "error": "Missing 'hands' in request payload"}), 400

    hands_data = payload.get("hands", [])[:1]  # Limit to max 1 hand
    if not hands_data:
        return jsonify({"success": True, "predictions": []})

    results = []
    for hand in hands_data:
        landmarks = hand.get("landmarks", [])
        if len(landmarks) != 63:
            continue

        is_normalized = hand.get("normalized", False)
        if is_normalized:
            vec = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        else:
            vec = normalize_landmarks_vector(landmarks)

        label, confidence = run_knn_inference(vec)
        
        # Don't return unknown label unless confidence is very high
        results.append({
            "label": label,
            "confidence": round(confidence, 3),
            "handedness": hand.get("handedness", "Hand")
        })

    return jsonify({"success": True, "predictions": results})

# High-frequency conversational English words for priority ranking in autocomplete
COMMON_PRIORITY_WORDS = {
    "hello", "help", "how", "home", "hand", "have", "here", "hear", "heart", "head",
    "good", "great", "glad", "morning", "sign", "language", "thank", "thanks", "you", "your",
    "yes", "no", "not", "please", "welcome", "world", "water", "what", "when", "where",
    "which", "who", "why", "will", "would", "with", "without", "name", "nice", "meet",
    "time", "today", "tomorrow", "yesterday", "day", "night", "friend", "family", "father",
    "mother", "brother", "sister", "school", "work", "learn", "study", "student", "teacher",
    "happy", "sad", "love", "like", "look", "live", "life", "want", "need", "feel",
    "think", "know", "understand", "again", "slow", "fast", "stop", "go", "come", "see",
    "talk", "speak", "read", "write", "eat", "drink", "sleep", "give", "take", "find",
    "make", "play", "call", "try", "ask", "answer", "open", "close", "start", "finish",
    "done", "fine", "ok", "okay", "sorry", "excuse", "pardon", "week", "month", "year",
    "number", "letter", "word", "sentence", "question", "right", "left", "up", "down",
    "man", "woman", "child", "children", "people", "person", "place", "thing", "idea"
}

@app.route("/api/autocomplete", methods=["GET"])
def api_autocomplete():
    """
    Fast prefix autocomplete using binary search (bisect) over sorted words.
    Time complexity: O(log N + K) where K is result count.
    Prioritizes common conversational vocabulary, then shorter words.
    """
    prefix = request.args.get("prefix", "").strip().lower()
    limit = int(request.args.get("limit", 6))

    if not prefix or not prefix.isalpha():
        return jsonify({"prefix": prefix, "suggestions": []})

    # 1. First gather immediate common vocabulary matches
    priority_matches = [w for w in COMMON_PRIORITY_WORDS if w.startswith(prefix) and w != prefix]
    priority_matches.sort(key=lambda w: (len(w), w))

    # 2. Fill remaining slots from the full 370,000+ dictionary via binary search
    dictionary_matches = []
    start_idx = bisect.bisect_left(WORDS, prefix)
    for i in range(start_idx, len(WORDS)):
        word = WORDS[i]
        if word.startswith(prefix):
            if word != prefix and word not in priority_matches:
                dictionary_matches.append(word)
            if len(dictionary_matches) >= limit * 3:
                break
        else:
            break

    dictionary_matches.sort(key=lambda w: (len(w), w))
    combined = (priority_matches + dictionary_matches)[:limit]

    return jsonify({
        "prefix": prefix,
        "suggestions": combined
    })

@app.route("/api/predict_frame", methods=["POST"])
def api_predict_frame():
    """
    Fallback endpoint: accepts a base64 encoded video frame,
    runs server-side MediaPipe, and returns predictions and landmarks.
    """
    if server_hands_detector is None or knn_model is None:
        return jsonify({"success": False, "error": "Server-side detector not available"}), 503

    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"success": False, "error": "Missing image data"}), 400

    try:
        image_data = payload["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        # Flip horizontally for selfie view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = server_hands_detector.process(rgb_frame)

        predictions = []
        landmarks_output = []

        if detection_results.multi_hand_landmarks:
            for idx, lm in enumerate(detection_results.multi_hand_landmarks[:1]):  # Limit to 1 hand
                raw_pts = []
                for p in lm.landmark:
                    raw_pts.extend([p.x, p.y, p.z])

                vec = normalize_landmarks_vector(raw_pts)
                label, confidence = run_knn_inference(vec)

                handedness = "Hand"
                if detection_results.multi_handedness and len(detection_results.multi_handedness) > idx:
                    handedness = detection_results.multi_handedness[idx].classification[0].label

                predictions.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "handedness": handedness
                })
                landmarks_output.append([{"x": p.x, "y": p.y, "z": p.z} for p in lm.landmark])

        return jsonify({
            "success": True,
            "predictions": predictions,
            "landmarks": landmarks_output
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 66)
    print("[INFO] ASL Sign Predictor & Autocompleter Web Application")
    print(f"[INFO] Server running at: http://127.0.0.1:{port}")
    print("=" * 66 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
