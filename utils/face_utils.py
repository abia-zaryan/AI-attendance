import os
import tempfile
import threading
from datetime import datetime

import cv2
import numpy as np

from config import FACE_SIZE

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceEngine:
    def __init__(self, threshold=60):
        self.threshold = threshold
        self.detector = cv2.CascadeClassifier(_CASCADE_PATH)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=threshold)
        self.labels = {}        
        self._trained = False
        self._lock = threading.Lock()


    def detect_faces(self, gray):
        return self.detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

    def _largest_face(self, gray):
        faces = self.detect_faces(gray)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _prepare(self, gray, x, y, w, h):
        face = gray[y:y + h, x:x + w]
        face = cv2.equalizeHist(face)
        return cv2.resize(face, FACE_SIZE)


    def save_sample(self, db, student_id, bgr_image):
        """Detect the face in a frame and store it as a Mongo document.
        Returns (True, new_count) or (False, current_count)."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        box = self._largest_face(gray)
        current = self.sample_count(db, student_id)
        if box is None:
            return False, current
        x, y, w, h = box
        face = self._prepare(gray, x, y, w, h)
        ok, buf = cv2.imencode(".jpg", face)
        if not ok:
            return False, current
        seq = current + 1
        db.face_samples.insert_one({
            "student_id": student_id,
            "seq": seq,
            "image": buf.tobytes(),
            "created_at": datetime.now(),
        })
        return True, seq

    def sample_count(self, db, student_id):
        return db.face_samples.count_documents({"student_id": student_id})

    def delete_student_samples(self, db, student_id):
        db.face_samples.delete_many({"student_id": student_id})


    def train(self, db):
        faces, labels = [], []
        label_map = {}
        for sid in db.face_samples.distinct("student_id"):
            cursor = db.face_samples.find({"student_id": sid})
            label = None
            for doc in cursor:
                img = cv2.imdecode(np.frombuffer(doc["image"], dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if label is None:
                    label = len(label_map)
                    label_map[label] = sid
                faces.append(cv2.resize(img, FACE_SIZE))
                labels.append(label)

        if not faces:
            self.labels = {}
            self._trained = False
            db.face_model.delete_one({"_id": "current"})
            return {"success": False,
                    "message": "No face samples found. Capture samples for at least one student first."}

        with self._lock:
            recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=self.threshold)
            recognizer.train(faces, np.array(labels))
            tmp_path = os.path.join(tempfile.gettempdir(), "lbph_model.xml")
            recognizer.write(tmp_path)
            with open(tmp_path, "rb") as f:
                model_bytes = f.read()
            os.remove(tmp_path)

            db.face_model.update_one(
                {"_id": "current"},
                {"$set": {
                    "model": model_bytes,
                    "labels": {str(k): v for k, v in label_map.items()},
                    "updated_at": datetime.now(),
                }},
                upsert=True,
            )
            self.recognizer = recognizer
            self.labels = label_map
            self._trained = True

        return {"success": True,
                "message": f"Model trained on {len(label_map)} student(s), {len(faces)} image(s)."}

    def load_model(self, db):
        """Load the trained model from MongoDB into memory. Call at startup
        and it's safe to call again any time (e.g. after a redeploy)."""
        doc = db.face_model.find_one({"_id": "current"})
        if not doc or not doc.get("model"):
            self.labels = {}
            self._trained = False
            return False
        tmp_path = os.path.join(tempfile.gettempdir(), "lbph_model_load.xml")
        with open(tmp_path, "wb") as f:
            f.write(doc["model"])
        recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=self.threshold)
        try:
            recognizer.read(tmp_path)
        except cv2.error:
            os.remove(tmp_path)
            self.labels = {}
            self._trained = False
            return False
        os.remove(tmp_path)
        with self._lock:
            self.recognizer = recognizer
            self.labels = {int(k): v for k, v in doc.get("labels", {}).items()}
            self._trained = True
        return True

    def is_trained(self):
        return self._trained and bool(self.labels)


    def recognize(self, bgr_image):
        """Return a list of {student_id, confidence, box} for every detected
        face. student_id is None for a face that doesn't match anyone."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detections = self.detect_faces(gray)
        results = []
        with self._lock:
            for (x, y, w, h) in detections:
                face = self._prepare(gray, x, y, w, h)
                label, distance = self.recognizer.predict(face)
                known = (label != -1 and label in self.labels
                         and distance <= self.threshold)
                results.append({
                    "student_id": self.labels[label] if known else None,
                    "confidence": round(100 - (distance / self.threshold) * 100, 1) if known else 0,
                    "box": [int(x), int(y), int(w), int(h)],
                })
        return results
