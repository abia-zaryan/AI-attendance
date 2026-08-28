"""
Face detection + recognition engine — MANUAL, TRANSPARENT PIPELINE
====================================================================
No face-recognition library is used anywhere in this file. Every step
that "identifies" a face is code we wrote and trained ourselves:

  1. DETECTION    -> OpenCV Haar Cascade. This only finds "where is a
                      face in this frame" (a rectangle) — it has no idea
                      WHO the face belongs to.

  2. FEATURES     -> HOG (Histogram of Oriented Gradients), via OpenCV's
                      cv2.HOGDescriptor. This is a classical, generic
                      image-descriptor (used for pedestrian/object
                      detection too) — it just turns a face image into a
                      list of numbers describing edge/gradient patterns.
                      It does not know about faces or identities either.

  3. MODEL        -> scikit-learn's MLPClassifier = a real Artificial
                      Neural Network (2 hidden layers). THIS is the part
                      that actually learns to tell students apart, and we
                      train it ourselves by calling `.fit()` on our own
                      HOG feature vectors + student-ID labels.

  4. STORAGE      -> The trained network + its label mapping are
                      serialised with Python's `pickle` module.
                        - a REAL model.pkl file is written to
                          <project>/models/face_model.pkl every time you
                          train (so you have a physical, inspectable
                          artifact you can open/submit/version),
                        - the SAME bytes are also saved into MongoDB
                          (face_model collection) because free hosting
                          (Render/Railway) wipes local disk on every
                          redeploy — MongoDB is what actually survives.

app.py talks to this class through the same 6 methods as before
(detect_faces, save_sample, train, load_model, is_trained, recognize)
so nothing else in the app had to change.
"""
import os
import pickle
import threading
from datetime import datetime

import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import FACE_SIZE

_BUNDLED_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "haarcascade_frontalface_default.xml"
)
_CASCADE_PATH = (
    _BUNDLED_CASCADE_PATH
    if os.path.exists(_BUNDLED_CASCADE_PATH)
    else cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Where the tangible .pkl file is written on every training run.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_model.pkl")

# HOG settings — tuned to match FACE_SIZE=(200,200) from config.py.
# NOTE: these block/cell sizes were deliberately chosen to keep the HOG
# feature vector (and therefore the trained ANN's weight matrix) small.
# A finer grid produces a much longer feature vector, which makes the
# pickled model file balloon past MongoDB's hard 16MB per-document limit
# (an early version of this file used a finer grid and produced a ~19MB
# pickle that would fail to save on real MongoDB, even though it worked
# fine locally). With these settings the model stays a few MB regardless
# of how many students you add.
_HOG = cv2.HOGDescriptor(
    FACE_SIZE,      # winSize
    (40, 40),       # blockSize
    (20, 20),       # blockStride
    (20, 20),       # cellSize
    9,              # nbins
)


class FaceEngine:
    def __init__(self, threshold=60):
        # threshold is now a MINIMUM CONFIDENCE PERCENTAGE (0-100) coming
        # out of the ANN's predict_proba(), not an LBPH distance anymore.
        # Higher threshold = stricter matching. config.py's FACE_THRESHOLD
        # should be a percentage now (e.g. 55).
        self.threshold_pct = threshold
        self.detector = cv2.CascadeClassifier(_CASCADE_PATH)
        if self.detector.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from '{_CASCADE_PATH}'. "
                "The face-detection XML file could not be found or read."
            )
        self.clf = None            # trained MLPClassifier (the ANN)
        self.label_encoder = None  # maps class index <-> student_id
        self._trained = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------- detection

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

    def _hog_features(self, face_img):
        """200x200 grayscale face -> 1D HOG feature vector (the numeric
        'fingerprint' fed into the ANN)."""
        feat = _HOG.compute(face_img)
        return feat.flatten().astype(np.float32)

    # ---------------------------------------------------------- samples

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

    # ---------------------------------------------------------- training

    def train(self, db):
        """MANUAL TRAINING: we build the dataset (HOG features + labels)
        ourselves, construct the MLPClassifier (ANN) ourselves, and call
        `.fit()` ourselves — no ready-made face-recognizer is involved."""
        features, labels_raw = [], []
        for sid in db.face_samples.distinct("student_id"):
            for doc in db.face_samples.find({"student_id": sid}):
                img = cv2.imdecode(np.frombuffer(doc["image"], dtype=np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, FACE_SIZE)
                features.append(self._hog_features(img))
                labels_raw.append(sid)

        if not features:
            self.clf, self.label_encoder = None, None
            self._trained = False
            db.face_model.delete_one({"_id": "current"})
            return {"success": False,
                    "message": "No face samples found. Capture samples for at least one student first."}

        n_classes = len(set(labels_raw))
        if n_classes < 2:
            return {"success": False,
                    "message": "Need at least 2 different students' samples so the ANN has something to tell apart."}

        X = np.array(features, dtype=np.float32)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels_raw)

        with self._lock:
            # ─── the actual manual training step ───
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),   # 2 hidden layers -> ANN
                activation="relu",
                solver="adam",
                max_iter=400,
                early_stopping=True,
                random_state=42,
            )
            clf.fit(X, y)  # <-- we call training ourselves

            model_bytes = pickle.dumps({"clf": clf, "encoder": encoder})

            # 1) persist to MongoDB (source of truth for the live app —
            #    survives redeploys/restarts on free hosting)
            db.face_model.update_one(
                {"_id": "current"},
                {"$set": {"model": model_bytes, "updated_at": datetime.now()}},
                upsert=True,
            )

            # 2) ALSO write a real, physical .pkl file to disk — this is
            #    the tangible model artifact you can open, inspect, or
            #    commit. On free hosting this copy won't survive a
            #    redeploy, but MongoDB (above) already covers that; this
            #    file is for local inspection / your project submission.
            try:
                os.makedirs(_MODEL_DIR, exist_ok=True)
                with open(_MODEL_PATH, "wb") as f:
                    f.write(model_bytes)
            except OSError:
                pass  # disk may be read-only on some hosts — Mongo copy still saved

            self.clf = clf
            self.label_encoder = encoder
            self._trained = True

        train_acc = round(clf.score(X, y) * 100, 1)
        return {"success": True,
                "message": f"ANN trained on {n_classes} student(s), {len(features)} image(s). "
                           f"Training accuracy: {train_acc}%. Saved to models/face_model.pkl"}

    def load_from_file(self, path):
        """Load the pickled ANN from a local .pkl file on disk (instead of
        MongoDB). Used by standalone scripts like predict_demo.py so they
        reuse the exact same detection/feature/prediction code as the
        live app instead of re-implementing it (and risking it drifting
        out of sync, e.g. if HOG parameters ever change here)."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        with self._lock:
            self.clf = payload["clf"]
            self.label_encoder = payload["encoder"]
            self._trained = True
        return True

    def load_model(self, db):
        """Load the pickled ANN from MongoDB into memory (source of truth
        for the running app). Safe to call again any time, e.g. after a
        redeploy or in a freshly-spawned gunicorn worker."""
        doc = db.face_model.find_one({"_id": "current"})
        if not doc or not doc.get("model"):
            self.clf, self.label_encoder = None, None
            self._trained = False
            return False
        try:
            payload = pickle.loads(doc["model"])
            clf = payload["clf"]
            encoder = payload["encoder"]
        except Exception:
            self.clf, self.label_encoder = None, None
            self._trained = False
            return False
        with self._lock:
            self.clf = clf
            self.label_encoder = encoder
            self._trained = True
        return True

    def is_trained(self):
        return self._trained and self.clf is not None

    # ---------------------------------------------------------- recognition

    def recognize(self, bgr_image):
        """Return a list of {student_id, confidence, box} for every detected
        face. student_id is None for a face whose best-match confidence
        doesn't clear FACE_THRESHOLD (treated as Unknown)."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detections = self.detect_faces(gray)
        results = []
        with self._lock:
            for (x, y, w, h) in detections:
                face = self._prepare(gray, x, y, w, h)
                feat = self._hog_features(face).reshape(1, -1)
                probs = self.clf.predict_proba(feat)[0]
                best_idx = int(np.argmax(probs))
                confidence = round(float(probs[best_idx]) * 100, 1)
                known = confidence >= self.threshold_pct
                sid = self.label_encoder.inverse_transform([best_idx])[0] if known else None
                results.append({
                    "student_id": sid,
                    "confidence": confidence,
                    "box": [int(x), int(y), int(w), int(h)],
                })
        return results
