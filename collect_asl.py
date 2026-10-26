import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark_knn.joblib")
DATASET_PATH = os.path.join(BASE_DIR, "asl_landmarks_dataset.joblib")

def extract_landmarks(lm):
    """Normalize 21 hand landmarks to 63-dim vector matching training pipeline."""
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    pts[:, :2] -= pts[0, :2]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 1e-6:
        pts[:, :2] /= scale
    else:
        pts[:, :2] = 0
    return pts.flatten()

def load_existing_data():
    """Load existing training samples from dataset file or extract from trained KNN model."""
    if os.path.exists(DATASET_PATH):
        try:
            data = joblib.load(DATASET_PATH)
            X = list(data.get("X", []))
            y = list(data.get("y", []))
            print(f"[+] Loaded {len(X)} samples from {os.path.basename(DATASET_PATH)}")
            return X, y
        except Exception as e:
            print(f"[!] Warning reading dataset cache: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            knn = data["knn"]
            if hasattr(knn, "_fit_X") and hasattr(knn, "_y"):
                X = list(knn._fit_X)
                classes = list(knn.classes_)
                y = [str(classes[idx]) for idx in knn._y]
                print(f"[+] Loaded {len(X)} base samples from {os.path.basename(MODEL_PATH)}")
                return X, y
        except Exception as e:
            print(f"[!] Warning reading base model: {e}")

    return [], []

def train_and_save(X, y):
    """Retrain KNN on accumulated samples and save both model and dataset cache."""
    if len(X) == 0:
        print("No data collected yet!")
        return False

    X_arr = np.array(X)
    y_arr = np.array(y)
    labels = sorted(list(set(y)))
    
    print(f"\n[+] Training KNeighborsClassifier on {len(X_arr)} samples across {len(labels)} classes...")
    counts = Counter(y)
    for lab in labels:
        print(f"    - '{lab}': {counts[lab]} samples")

    k = min(5, len(X_arr))
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(X_arr, y_arr)

    # Save trained model compatible with app.py & asl_validation.py
    to_save = {"pca": None, "knn": knn, "labels": labels}
    joblib.dump(to_save, MODEL_PATH)

    # Save dataset backup for future incremental additions
    joblib.dump({"X": X, "y": y}, DATASET_PATH)

    print(f"[✓] Model saved successfully to: {MODEL_PATH}")
    print(f"[✓] Dataset cache saved to: {DATASET_PATH}\n")
    return True

def main():
    X, y = load_existing_data()
    counts = Counter(y)

    print("=" * 60)
    print(" 🤟 ASL Live Webcam Landmark Collector & Re-Trainer")
    print("=" * 60)
    print("Instructions:")
    print(" 1. Enter the target letter (e.g. 'z', 'd', 'q', etc.).")
    print(" 2. Position your hand in the webcam frame.")
    print(" 3. Press SPACEBAR to record landmark samples.")
    print(" 4. Press 'n' to switch to another letter.")
    print(" 5. Press 't' to train & save hand_landmark_knn.joblib.")
    print(" 6. Press 'c' to clear newly recorded samples for current letter.")
    print(" 7. Press 'q' to quit.")
    print("=" * 60)

    # Initialize current KNN model for real-time inference feedback
    live_knn = None
    if os.path.exists(MODEL_PATH):
        try:
            m_data = joblib.load(MODEL_PATH)
            live_knn = m_data.get("knn")
        except Exception:
            pass

    target_letter = input("Enter letter label to train/record (e.g., 'z'): ").strip().lower()
    if not target_letter:
        target_letter = "z"

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Error: Cannot access webcam. Check camera connection/permissions.")
        return

    recording = False
    samples_to_record = 40
    current_recorded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to grab camera frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            live_pred = "No Hand"
            lm_vec = None

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                lm_vec = extract_landmarks(lm)

                if live_knn is not None:
                    try:
                        live_pred = str(live_knn.predict([lm_vec])[0])
                    except Exception:
                        pass

                # If recording mode is triggered, save landmark sample
                if recording and current_recorded < samples_to_record:
                    X.append(lm_vec)
                    y.append(target_letter)
                    current_recorded += 1
                    if current_recorded >= samples_to_record:
                        recording = False
                        total_for_target = sum(1 for label in y if label == target_letter)
                        print(f"[✓] Finished batch! Total samples for '{target_letter}': {total_for_target}")

            # Draw HUD
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 85), (20, 20, 20), -1)
            frame = cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0)

            total_target = sum(1 for label in y if label == target_letter)
            rec_status = f"RECORDING ({current_recorded}/{samples_to_record})" if recording else "IDLE"
            rec_color = (0, 0, 255) if recording else (200, 200, 200)

            cv2.putText(frame, f"Target: '{target_letter.upper()}' | Samples: {total_target} | Status: {rec_status}", 
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)
            cv2.putText(frame, f"Live Model Prediction: {live_pred}", 
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if live_pred == target_letter else (0, 165, 255), 2)
            cv2.putText(frame, "[SPACE]: Record 40 | [N]: Next Letter | [T]: Train & Save | [Q]: Quit", 
                        (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("ASL Landmark Collector & Re-Trainer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if lm_vec is not None:
                    recording = True
                    current_recorded = 0
                    print(f"\n[!] Recording {samples_to_record} samples for '{target_letter}'... Keep signing with slight movements!")
                else:
                    print("[!] No hand detected. Please position your hand in frame before recording.")

            elif key == ord('n'):
                recording = False
                new_letter = input("\nEnter next letter label (e.g. 'd', 'z', 'q'): ").strip().lower()
                if new_letter:
                    target_letter = new_letter
                    total_target = sum(1 for label in y if label == target_letter)
                    print(f"[*] Switched target to '{target_letter}' ({total_target} samples currently).")

            elif key == ord('t'):
                recording = False
                success = train_and_save(X, y)
                if success:
                    # Reload live knn in memory
                    m_data = joblib.load(MODEL_PATH)
                    live_knn = m_data.get("knn")
                    print("[*] Live predictor updated with new weights!")

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    choice = input("\nTrain and save model before exiting? (y/n): ").strip().lower()
    if choice == 'y':
        train_and_save(X, y)

if __name__ == "__main__":
    main()
