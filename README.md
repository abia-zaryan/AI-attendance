# Attendance AI — Face Recognition Attendance Manager

A complete, working attendance system: Flask + MongoDB + OpenCV (LBPH face
recognition), with a live camera scanner, admin dashboard, reports, and CSV
export. Built to run locally **and** deploy for free without breaking.

## Why this survives free hosting

Free hosts (Render, Railway, PythonAnywhere) wipe the local disk on every
redeploy or restart. Most tutorials save captured face images and the
trained model as files — which means the whole face database disappears
the moment you redeploy. This project stores everything that matters
**inside MongoDB itself** (face samples + the trained model as bytes), so
nothing is lost when the app restarts. Local disk is only ever used as
scratch space.

It also uses `opencv-contrib-python-headless` instead of the normal
`opencv-contrib-python` — the non-headless build is the #1 cause of
`ImportError: libGL.so.1` crashes on Linux servers that have no display.

## Features

- Admin login (session based)
- Student registration + webcam face sample capture (in-browser, no upload needed)
- LBPH face model training, stored in MongoDB
- Live camera attendance: **Mark IN** / **Mark OUT** with on-screen bounding boxes
- One-record-per-student-per-day (basic proxy prevention) + "Finalize" auto-marks absentees
- Dashboard with 7-day trend + today's class breakdown (Chart.js)
- Records page with date/class/status filters
- Reports: per-student attendance %, low-attendance flag (<75%), CSV export

## Project structure

```
ai-attendance-manager/
├── app.py                  # Flask routes
├── config.py                # env-based settings
├── requirements.txt
├── Procfile                 # gunicorn start command (for hosting)
├── runtime.txt               # pinned Python version
├── .env.example
├── utils/
│   ├── db.py                # MongoDB connection helpers
│   └── face_utils.py         # face detection/recognition (Mongo-backed)
├── templates/                # all pages (Bootstrap 5 + Chart.js via CDN)
└── static/uploads/           # optional student profile photos
```

## 1. Run it locally

**Requirements:** Python 3.10–3.12, and a MongoDB connection (local install
*or* a free MongoDB Atlas cluster — Atlas is recommended even for local dev,
since you'll need it for hosting anyway).

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your `MONGO_URI` (skip this if
you're using a local MongoDB on the default port — the app falls back to
`mongodb://localhost:27017/` automatically). Then:

```bash
python app.py
```

Open **http://127.0.0.1:5000** — login with `admin` / `admin123` (or whatever
you set in `.env`).

### Getting a free MongoDB (2 minutes, recommended)

1. Go to mongodb.com/cloud/atlas → create a free account → create a free **M0** cluster.
2. Database Access → add a user with a password.
3. Network Access → add IP `0.0.0.0/0` (allow from anywhere — fine for a class project).
4. Connect → Drivers → copy the connection string, it looks like:
   `mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/`
5. Put that in `MONGO_URI` (in `.env` locally, and later as an environment
   variable on your hosting provider).

## 2. Push to GitHub

```bash
cd ai-attendance-manager
git init
git add .
git commit -m "Initial commit: AI Attendance Manager"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.gitignore` already excludes `venv/`, `.env`, and uploaded photos, so your
MongoDB password and virtual environment never get committed.

## 3. Deploy for free (Render.com)

Render's free web service tier works well for this project.

1. Push the repo to GitHub (step 2 above).
2. On render.com → **New +** → **Web Service** → connect your GitHub repo.
3. Settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** *(leave blank — Render reads it from `Procfile`)*
4. Add environment variables (Render dashboard → Environment):
   - `MONGO_URI` = your Atlas connection string
   - `SECRET_KEY` = any long random string
   - `ADMIN_USERNAME`, `ADMIN_PASSWORD` = your choice
   - `FLASK_DEBUG` = `0`
5. Deploy. Render gives you an HTTPS URL — the webcam works fine because
   it's served over HTTPS (browsers block camera access on plain HTTP).

**Alternatives:** Railway.app and PythonAnywhere work the same way — set
the same environment variables and use `gunicorn app:app` as the start
command.

### Free-tier note

Render's free web service **sleeps after inactivity** and takes ~30-50
seconds to wake up on the next visit — that's normal, not a bug. Because
face samples and the trained model live in MongoDB (not on disk), you will
**not** need to retrain the model after it wakes up or redeploys.

## 4. Using the app

1. Log in.
2. **Register Student** → fill details → you're taken straight to face capture.
3. **Capture Faces** → click *Start Camera* → *Auto Capture* → look
   straight, then slightly left/right/up/down until you hit the target
   (40 is enough; raise it to 100+ for higher accuracy).
4. Repeat for every student, then click **Train Face Model** (Students
   page or the capture page).
5. **Take Attendance** → *Start Scanning* in **Mark IN** mode — recognized
   students are marked present automatically. Switch to **Mark OUT** at
   the end of class to log out-time and duration.
6. Click **Finalize** once, at the end of the day, to mark everyone who
   was never detected as **Absent**.
7. Check **Records** (filter by date/class/status, export CSV) and
   **Reports** (per-student % with a low-attendance flag under 75%).

## Troubleshooting

| Problem | Fix |
|---|---|
| `cv2.face not found` | `pip uninstall opencv-python opencv-python-headless -y` then `pip install opencv-contrib-python-headless` |
| `ServerSelectionTimeoutError` | MongoDB isn't reachable — check `MONGO_URI`, and on Atlas confirm Network Access allows your IP |
| Camera is black / permission denied | Use Chrome or Edge, allow the camera prompt, close other apps using the camera. Camera access requires `localhost` or HTTPS — it will **not** work over plain `http://` on a public host |
| Port 5000 already in use | `set PORT=5001` (Windows) or `PORT=5001 python app.py` (macOS/Linux) |
| Recognition is inaccurate | Capture more samples (100+) with varied angles/lighting, then retrain; lower `FACE_THRESHOLD` in `.env` for stricter matching |
| Face data "disappeared" after redeploy | Confirm `MONGO_URI` points to Atlas (not local Mongo) — everything is stored there by design |

## Tech stack

Flask · MongoDB (PyMongo) · OpenCV (Haar Cascade detection + LBPH
recognition) · Bootstrap 5 · Chart.js · gunicorn

---

Built for a college/institute attendance project. Not intended for
high-security or large-scale (1000+ student) deployments — LBPH is a
lightweight classical recognizer, ideal for classroom-scale accuracy.
