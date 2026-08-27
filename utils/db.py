from datetime import datetime

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.security import generate_password_hash

from config import DB_NAME, DEFAULT_ADMIN_PASS, DEFAULT_ADMIN_USER, MONGO_URI

_client = None


def get_client():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    return _client


def get_db():
    return get_client()[DB_NAME]


def ping():
    """Return True if MongoDB is reachable right now."""
    try:
        get_client().admin.command("ping")
        return True
    except PyMongoError:
        return False


def init_db():
    """Create the default admin user + indexes. Safe to call repeatedly."""
    db = get_db()
    if db.users.count_documents({"username": DEFAULT_ADMIN_USER}) == 0:
        db.users.insert_one({
            "username": DEFAULT_ADMIN_USER,
            "password_hash": generate_password_hash(DEFAULT_ADMIN_PASS),
            "role": "admin",
            "created_at": datetime.now(),
        })
    try:
        db.students.create_index("student_id", unique=True)
        db.attendance.create_index([("student_id", 1), ("date", 1)])
        db.face_samples.create_index([("student_id", 1), ("seq", 1)])
    except PyMongoError:
        pass


def today_str():
    return datetime.now().strftime("%Y-%m-%d")
