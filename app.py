import base64
import csv
import io
import os
import re
from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                    send_file, session, url_for)
from werkzeug.security import check_password_hash

from config import FACE_THRESHOLD, SECRET_KEY, UPLOAD_DIR
from utils.db import get_db, init_db, ping, today_str
from utils.face_utils import FaceEngine

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "cv2.face not found. Install the contrib build:\n"
        "  pip uninstall -y opencv-python opencv-python-headless\n"
        "  pip install opencv-contrib-python-headless"
    )

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.permanent_session_lifetime = timedelta(hours=8)

face_engine = FaceEngine(threshold=FACE_THRESHOLD)
_model_load_attempted = False


def ensure_model_loaded():
    """Lazily pull the trained model from MongoDB on first use of a worker
    (gunicorn spawns multiple processes, each needs its own in-memory copy)."""
    global _model_load_attempted
    if not face_engine.is_trained() and ping():
        face_engine.load_model(get_db())
    _model_load_attempted = True



def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Please log in first."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def decode_image(data_url):
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        buf = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.template_filter("timefmt")
def timefmt(value):
    if isinstance(value, datetime):
        return value.strftime("%I:%M:%S %p")
    return value or "—"


@app.template_filter("durfmt")
def durfmt(minutes):
    if minutes is None:
        return "—"
    minutes = int(minutes)
    return f"{minutes // 60}h {minutes % 60}m"


@app.context_processor
def inject_globals():
    return {
        "now": datetime.now(),
        "mongo_ok": ping(),
        "trained": face_engine.is_trained(),
    }



@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))
    mongo_up = ping()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not mongo_up:
            flash("Cannot connect to MongoDB. Check your MONGO_URI and try again.", "danger")
            return render_template("login.html")
        user = get_db().users.find_one({"username": username})
        if user and check_password_hash(user["password_hash"], password):
            session.permanent = True
            session["user"] = username
            ensure_model_loaded()
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    elif not mongo_up:
        flash("Cannot connect to MongoDB. Check your MONGO_URI.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ───────────────────────────── dashboard ─────────────────────────────

@app.route("/")
@login_required
def dashboard():
    db = get_db()
    t = today_str()
    total_students = db.students.count_documents({})
    present_today = db.attendance.count_documents({"date": t, "status": "present"})
    absent_today = db.attendance.count_documents({"date": t, "status": "absent"})
    rate = round(present_today / total_students * 100, 1) if total_students else 0.0

    trend = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        trend.append({"date": d[5:], "present": db.attendance.count_documents(
            {"date": d, "status": "present"})})

    class_counts = list(db.attendance.aggregate([
        {"$match": {"date": t, "status": "present"}},
        {"$group": {"_id": {"$ifNull": ["$class_name", "N/A"]}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]))

    recent = list(db.attendance.find({"date": t}).sort("created_at", -1).limit(8))

    return render_template("dashboard.html", total_students=total_students,
                           present_today=present_today, absent_today=absent_today,
                           rate=rate, trend=trend, class_counts=class_counts,
                           recent=recent)




@app.route("/students")
@login_required
def students():
    db = get_db()
    all_students = list(db.students.find({}).sort("created_at", -1))
    for s in all_students:
        s["samples"] = face_engine.sample_count(db, s["student_id"])
        s["present_count"] = db.attendance.count_documents(
            {"student_id": s["student_id"], "status": "present"})
        s["edit_data"] = {
            "student_id": s["student_id"], "name": s["name"],
            "roll_no": s.get("roll_no", ""), "class_name": s.get("class_name", ""),
            "email": s.get("email", ""), "phone": s.get("phone", ""),
        }
    return render_template("students.html", students=all_students)


@app.route("/students/register", methods=["GET", "POST"])
@login_required
def register_student():
    if request.method == "POST":
        db = get_db()
        student_id = request.form.get("student_id", "").strip().upper()
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        class_name = request.form.get("class_name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        if not re.fullmatch(r"[A-Za-z0-9_-]{2,30}", student_id):
            flash("Student ID must be 2-30 characters: letters, numbers, - or _ only.", "danger")
            return redirect(url_for("register_student"))
        if not name:
            flash("Name is required.", "danger")
            return redirect(url_for("register_student"))
        if db.students.find_one({"student_id": student_id}):
            flash(f"Student ID '{student_id}' already exists.", "danger")
            return redirect(url_for("register_student"))

        photo = None
        file = request.files.get("photo")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp"):
                filename = f"{student_id}{ext}"
                file.save(os.path.join(UPLOAD_DIR, filename))
                photo = f"uploads/{filename}"

        db.students.insert_one({
            "student_id": student_id, "name": name, "roll_no": roll_no,
            "class_name": class_name, "email": email, "phone": phone,
            "photo": photo, "created_at": datetime.now(),
        })
        flash(f"{name} registered. Now capture face samples.", "success")
        return redirect(url_for("capture_faces", student_id=student_id))

    return render_template("register_student.html")


@app.route("/students/update/<student_id>", methods=["POST"])
@login_required
def update_student(student_id):
    db = get_db()
    db.students.update_one({"student_id": student_id}, {"$set": {
        "name": request.form.get("name", "").strip(),
        "roll_no": request.form.get("roll_no", "").strip(),
        "class_name": request.form.get("class_name", "").strip(),
        "email": request.form.get("email", "").strip(),
        "phone": request.form.get("phone", "").strip(),
    }})
    flash("Student details updated.", "success")
    return redirect(url_for("students"))


@app.route("/students/delete/<student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    db = get_db()
    db.students.delete_one({"student_id": student_id})
    db.attendance.delete_many({"student_id": student_id})
    face_engine.delete_student_samples(db, student_id)
    result = face_engine.train(db)
    msg = result["message"] if result["success"] else "Retrain the model from the Students page."
    flash(f"Student '{student_id}' deleted. {msg}", "success")
    return redirect(url_for("students"))


@app.route("/students/<student_id>/capture")
@login_required
def capture_faces(student_id):
    student = get_db().students.find_one({"student_id": student_id})
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("students"))
    samples = face_engine.sample_count(get_db(), student_id)
    return render_template("capture_faces.html", student=student, samples=samples)


@app.route("/students/<student_id>/reset-samples", methods=["POST"])
@login_required
def reset_samples(student_id):
    face_engine.delete_student_samples(get_db(), student_id)
    flash("Face samples cleared. Capture again and retrain the model.", "warning")
    return redirect(url_for("capture_faces", student_id=student_id))



@app.route("/api/capture-frame", methods=["POST"])
@login_required
def api_capture_frame():
    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id", "")
    img = decode_image(data.get("image", ""))
    if img is None:
        return jsonify({"success": False, "count": 0, "message": "Invalid image received."})
    db = get_db()
    if not db.students.find_one({"student_id": student_id}):
        return jsonify({"success": False, "count": 0, "message": "Student not found."})
    count = face_engine.sample_count(db, student_id)
    if count >= 200:
        return jsonify({"success": False, "count": count, "message": "Sample limit reached (200)."})
    ok, new_count = face_engine.save_sample(db, student_id, img)
    if ok:
        return jsonify({"success": True, "count": new_count, "message": None})
    return jsonify({"success": False, "count": count,
                    "message": "No face detected - adjust lighting/position."})


@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    result = face_engine.train(get_db())
    return jsonify(result)




@app.route("/attendance")
@login_required
def take_attendance():
    ensure_model_loaded()
    return render_template("take_attendance.html", today=today_str())


@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    if not face_engine.is_trained():
        ensure_model_loaded()
    if not face_engine.is_trained():
        return jsonify({"success": False, "message": "Model not trained yet."})

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "in")
    img = decode_image(data.get("image", ""))
    if img is None:
        return jsonify({"success": False, "message": "Invalid image."})

    db = get_db()
    t = today_str()
    now = datetime.now()
    detections = face_engine.recognize(img)

    faces, events, seen = [], [], set()
    for det in detections:
        sid = det["student_id"]
        student = db.students.find_one({"student_id": sid}) if sid else None
        if not student:
            faces.append({"student_id": None, "name": "Unknown", "confidence": 0, "box": det["box"]})
            continue
        faces.append({"student_id": sid, "name": student["name"],
                      "confidence": det["confidence"], "box": det["box"]})
        if sid in seen:
            continue
        seen.add(sid)

        record = db.attendance.find_one({"student_id": sid, "date": t})
        event = {"student_id": sid, "name": student["name"], "action": "", "detail": ""}

        if mode == "in":
            if record and record.get("status") == "present":
                event["action"], event["detail"] = "already", "Already marked present"
            else:
                db.attendance.update_one(
                    {"student_id": sid, "date": t},
                    {"$set": {"student_id": sid, "name": student["name"],
                              "class_name": student.get("class_name", ""), "date": t,
                              "in_time": now, "status": "present", "created_at": now},
                     "$setOnInsert": {"out_time": None, "duration_minutes": None}},
                    upsert=True)
                event["action"], event["detail"] = "in", f"Marked present at {now.strftime('%I:%M:%S %p')}"
        else:
            if record and record.get("in_time") and not record.get("out_time"):
                dur = int((now - record["in_time"]).total_seconds() // 60)
                db.attendance.update_one({"_id": record["_id"]},
                                         {"$set": {"out_time": now, "duration_minutes": max(dur, 0)}})
                event["action"], event["detail"] = "out", f"Out at {now.strftime('%I:%M:%S %p')} ({max(dur, 0)} min)"
            elif record and record.get("out_time"):
                event["action"], event["detail"] = "already", "Out-time already recorded"
            else:
                event["action"], event["detail"] = "none", "No in-time recorded for today"
        events.append(event)

    return jsonify({"success": True, "faces": faces, "events": events})


@app.route("/api/attendance/today")
@login_required
def api_today_attendance():
    db = get_db()
    records = list(db.attendance.find({"date": today_str()}).sort("created_at", -1))
    for r in records:
        r["_id"] = str(r["_id"])
        r["in_time"] = r["in_time"].strftime("%I:%M:%S %p") if r.get("in_time") else None
        r["out_time"] = r["out_time"].strftime("%I:%M:%S %p") if r.get("out_time") else None
    return jsonify({"success": True, "records": records})


@app.route("/api/attendance/finalize", methods=["POST"])
@login_required
def api_finalize():
    db = get_db()
    t = today_str()
    now = datetime.now()
    marked = {r["student_id"] for r in db.attendance.find({"date": t}, {"student_id": 1})}
    absent = []
    for s in db.students.find({}, {"student_id": 1, "name": 1, "class_name": 1}):
        if s["student_id"] in marked:
            continue
        db.attendance.insert_one({
            "student_id": s["student_id"], "name": s["name"],
            "class_name": s.get("class_name", ""), "date": t,
            "in_time": None, "out_time": None, "duration_minutes": None,
            "status": "absent", "created_at": now,
        })
        absent.append(s["name"])
    return jsonify({"success": True, "absent_count": len(absent), "absent_names": absent})



@app.route("/records")
@login_required
def records():
    db = get_db()
    d = request.args.get("date", "").strip()
    cls = request.args.get("class", "").strip()
    status = request.args.get("status", "").strip()
    q = {}
    if d:
        q["date"] = d
    if cls:
        q["class_name"] = cls
    if status:
        q["status"] = status
    classes = [c for c in db.students.distinct("class_name") if c]
    recs = list(db.attendance.find(q).sort([("date", -1), ("created_at", -1)]).limit(300))
    return render_template("records.html", records=recs, classes=classes,
                           filters={"date": d, "class": cls, "status": status})


def _parse_date(s, fallback):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return fallback


@app.route("/reports")
@login_required
def reports():
    db = get_db()
    end_d = _parse_date(request.args.get("end"), datetime.now())
    start_d = _parse_date(request.args.get("start"), datetime.now() - timedelta(days=6))
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    if (end_d - start_d).days > 90:
        start_d = end_d - timedelta(days=90)
    start, end = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")

    rows = []
    for s in db.students.find({}).sort("name", 1):
        total = db.attendance.count_documents(
            {"student_id": s["student_id"], "date": {"$gte": start, "$lte": end}})
        present = db.attendance.count_documents(
            {"student_id": s["student_id"], "date": {"$gte": start, "$lte": end}, "status": "present"})
        pct = round(present / total * 100, 1) if total else 0
        rows.append({"student_id": s["student_id"], "name": s["name"],
                     "class_name": s.get("class_name", ""), "present_days": present,
                     "absent_days": total - present, "total_days": total, "percentage": pct})

    daily = []
    d = start_d
    while d <= end_d:
        ds = d.strftime("%Y-%m-%d")
        daily.append({
            "date": ds[5:],
            "present": db.attendance.count_documents({"date": ds, "status": "present"}),
            "absent": db.attendance.count_documents({"date": ds, "status": "absent"}),
        })
        d += timedelta(days=1)

    total_present = sum(r["present_days"] for r in rows)
    total_marked = sum(r["total_days"] for r in rows)
    overall = round(total_present / total_marked * 100, 1) if total_marked else 0

    return render_template("reports.html", rows=rows, daily=daily, start=start, end=end,
                           overall=overall, total_present=total_present,
                           total_absent=total_marked - total_present)


@app.route("/export/csv")
@login_required
def export_csv():
    db = get_db()
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    q = {}
    if start or end:
        q["date"] = {}
        if start:
            q["date"]["$gte"] = start
        if end:
            q["date"]["$lte"] = end
    recs = list(db.attendance.find(q).sort([("date", 1), ("name", 1)]))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Student ID", "Name", "Class", "Date", "Status", "In Time", "Out Time", "Duration (minutes)"])
    for r in recs:
        writer.writerow([
            r.get("student_id"), r.get("name"), r.get("class_name", ""), r.get("date"), r.get("status", ""),
            r["in_time"].strftime("%H:%M:%S") if r.get("in_time") else "",
            r["out_time"].strftime("%H:%M:%S") if r.get("out_time") else "",
            r.get("duration_minutes") if r.get("duration_minutes") is not None else "",
        ])
    data = out.getvalue().encode("utf-8-sig")
    fname = f"attendance_{start or 'all'}_to_{end or 'all'}.csv"
    return send_file(io.BytesIO(data), as_attachment=True, download_name=fname, mimetype="text/csv")



@app.route("/health")
def health():
    return jsonify({"mongo": ping(), "model_trained": face_engine.is_trained()})


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    msg = "MongoDB connection failed. Check MONGO_URI and try again." if not ping() else "Something went wrong."
    return render_template("error.html", message=msg), 500


if __name__ == "__main__":
    print("=" * 62)
    print("  AI ATTENDANCE MANAGER  |  Flask + MongoDB + OpenCV (LBPH)")
    print("=" * 62)
    if ping():
        init_db()
        face_engine.load_model(get_db())
        print(f"[OK] MongoDB connected  ->  login: {os.environ.get('ADMIN_USERNAME','admin')} / "
              f"{os.environ.get('ADMIN_PASSWORD','admin123')}")
    else:
        print("[!!] MongoDB NOT reachable. Set MONGO_URI or start MongoDB locally.")
    port = int(os.environ.get("PORT", 5000))
    print(f"[OK] Open  http://127.0.0.1:{port}  in your browser")
    print("=" * 62)
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", threaded=True, host="0.0.0.0", port=port)
else:
    if ping():
        init_db()
        face_engine.load_model(get_db())
