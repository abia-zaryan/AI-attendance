"""
Standalone demo: loads models/face_model.pkl directly (no Flask, no
MongoDB) and identifies the face in a single image. This is the clearest
possible proof that "the pickle file identifies images" on its own —
everything underneath is plain OpenCV + scikit-learn, no face-recognition
library.

It reuses utils/face_utils.py's FaceEngine (the exact same detection +
HOG + ANN code the web app uses) instead of re-implementing any of that
logic here — that keeps this script guaranteed in sync with the app.

Usage:
    python predict_demo.py path/to/photo.jpg
"""
import sys

import cv2

from config import FACE_THRESHOLD
from utils.face_utils import FaceEngine

MODEL_PATH = "models/face_model.pkl"


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_demo.py path/to/photo.jpg")
        sys.exit(1)
    image_path = sys.argv[1]

    engine = FaceEngine(threshold=FACE_THRESHOLD)
    try:
        engine.load_from_file(MODEL_PATH)
    except FileNotFoundError:
        print(f"No trained model found at {MODEL_PATH}. "
              f"Train it first (via the app's 'Train Model' button, or `python train_model.py`).")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    results = engine.recognize(img)
    if not results:
        print("No face detected in this image.")
        return

    for r in results:
        if r["student_id"]:
            print(f"Match: {r['student_id']}  (confidence: {r['confidence']}%)")
        else:
            print(f"Unknown face (confidence {r['confidence']}%, "
                  f"below the {FACE_THRESHOLD}% threshold)")


if __name__ == "__main__":
    main()
