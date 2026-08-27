import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "ai_attendance_manager")

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-secret-change-this-in-production")
DEFAULT_ADMIN_USER = os.environ.get("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASS = os.environ.get("ADMIN_PASSWORD", "admin123")

FACE_THRESHOLD = int(os.environ.get("FACE_THRESHOLD", "60"))
FACE_SIZE = (200, 200)

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
