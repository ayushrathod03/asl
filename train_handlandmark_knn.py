import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ----------------------------
# Default Config
# ----------------------------
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_OUT_MODEL = os.path.join(BASE_DIR, "hand_landmark_knn.joblib")
USE_PCA = False          # set True to enable PCA if you have many dims / want to regularize
PCA_COMPONENTS = 30      # used only if USE_PCA = True
K_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_DETECTION_CONF = 0.1
DEFAULT_MAX_SAMPLES = 0  # Default max images per class (set None or 0 for all)
# ----------------------------

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True,
                                max_num_hands=1,
                                min_detection_confidence=MIN_DETECTION_CONF)

def extract_landmarks_from_image(img_bgr):
    """Return 63-dim vector (x,y,z) normalized: wrist as origin and scaled."""
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if h < 200 or w < 200:
        pad = max(h, w)
        padded = np.zeros((pad, pad, 3), dtype=np.uint8)
        padded[(pad - h) // 2 : (pad - h) // 2 + h, (pad - w) // 2 : (pad - w) // 2 + w] = img_bgr
        img_bgr = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])  # normalized positions
    # Translate so wrist (landmark 0) is origin (2D translation)
    pts[:, :2] -= pts[0, :2]
    # Scale by max 2D norm to be scale-invariant
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 1e-6:
        pts[:, :2] /= scale
    return pts.flatten()

def build_dataset(root, max_samples_per_class=DEFAULT_MAX_SAMPLES):
    X = []
    y = []
    # Alphabetical sorting of class subfolders
    labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not labels:
        raise ValueError(f"No label subfolders found in {root}. Create e.g. {root}\\a, {root}\\b, ...")
    print(f"Found {len(labels)} label folders in alphabetical order:")
    print(" ", labels)
    
    for lab in labels:
        folder = os.path.join(root, lab)
        count = 0
        file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if max_samples_per_class and max_samples_per_class > 0:
            file_list = file_list[:max_samples_per_class]
            
        for fname in tqdm(file_list, desc=f"Processing '{lab}'"):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            lm = extract_landmarks_from_image(img)
            if lm is not None:
                X.append(lm)
                y.append(lab)
                count += 1
        print(f"  --> Extracted {count} valid landmark samples for class '{lab}'")
    
    # Filter out any labels with < 2 samples to prevent train_test_split error
    X = np.array(X)
    y = np.array(y)
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        valid_labels = unique[counts >= 2]
        mask = np.isin(y, valid_labels)
        X = X[mask]
        y = y[mask]
        labels = sorted(list(set(y)))
    return X, y, labels

def main():
    parser = argparse.ArgumentParser(description="Train KNN on ASL dataset hand landmarks (alphabetical subfolders)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR, help="Path to dataset directory containing class subfolders")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_MODEL, help="Output path for trained KNN model file")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Max images per class to process (0 or negative for all)")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help="Number of neighbors for KNN classifier")
    parser.add_argument("--use_pca", action="store_true", default=USE_PCA, help="Enable PCA dimensionality reduction")
    parser.add_argument("--pca_components", type=int, default=PCA_COMPONENTS, help="Number of PCA components")
    args = parser.parse_args()

    dataset_root = args.dataset
    if not os.path.exists(dataset_root):
        fallback = r"D:\project\ASL-word-Autocompleter-and-Sign-Predictor-Agent-main\ASL-word-Autocompleter-and-Sign-Predictor-Agent-main\dataset"
        if os.path.exists(fallback):
            dataset_root = fallback
        else:
            raise FileNotFoundError(f"Dataset folder '{dataset_root}' not found!")

    max_samples = None if args.max_samples <= 0 else args.max_samples
    print("=" * 60)
    print("Starting ASL Hand Landmark KNN Training")
    print(f"Dataset Directory : {dataset_root}")
    print(f"Output Model Path : {args.out}")
    print(f"Max Samples/Class : {max_samples if max_samples else 'All available'}")
    print(f"KNN K-Neighbors   : {args.k}")
    print(f"PCA Enabled       : {args.use_pca}")
    print("=" * 60)

    X, y, labels = build_dataset(dataset_root, max_samples_per_class=max_samples)
    print("\nTotal extracted dataset shape (samples x dims):", X.shape)
    if X.shape[0] == 0:
        print("No landmarks extracted. Check images and lighting; ensure hands are visible.")
        return

    # Optional PCA
    pca = None
    if args.use_pca:
        print(f"Applying PCA -> {args.pca_components} components")
        pca = PCA(n_components=args.pca_components, random_state=RANDOM_STATE)
        X = pca.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Dataset split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    # Train KNN
    print("\nTraining KNeighborsClassifier...")
    knn = KNeighborsClassifier(n_neighbors=args.k, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nValidation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))

    # Save model and metadata
    to_save = {'pca': pca, 'knn': knn, 'labels': labels}
    joblib.dump(to_save, args.out)
    print(f"\nSuccessfully saved model and metadata to: {args.out}")

if __name__ == "__main__":
    main()
