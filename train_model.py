"""
Standalone CLI trainer — does exactly what the "Train Face Model" button
in the app does, but from the command line. Useful for retraining
without opening a browser, or for CI/scripts.

Usage:
    python train_model.py

Reads face samples from MongoDB (captured via the web app's camera
capture page), trains the HOG + MLPClassifier (ANN), and:
  1. saves the trained model into MongoDB (what the live Flask app uses)
  2. writes a real, physical file to models/face_model.pkl
"""
import sys

from utils.db import get_db, ping
from utils.face_utils import FaceEngine
from config import FACE_THRESHOLD


def main():
    if not ping():
        print("[!!] Cannot reach MongoDB. Check your MONGO_URI (see .env).")
        sys.exit(1)

    db = get_db()
    n_students = len(db.face_samples.distinct("student_id"))
    n_samples = db.face_samples.count_documents({})
    print(f"Found {n_samples} captured face sample(s) across {n_students} student(s).")

    engine = FaceEngine(threshold=FACE_THRESHOLD)
    result = engine.train(db)

    print()
    if result["success"]:
        print(f"[OK] {result['message']}")
        print("     -> models/face_model.pkl written")
        print("     -> also saved into MongoDB (face_model collection)")
    else:
        print(f"[!!] Training failed: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
