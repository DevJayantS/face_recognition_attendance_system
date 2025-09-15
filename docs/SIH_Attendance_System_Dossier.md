# Smart India Hackathon (SIH) – AI Attendance System Dossier

## 1. Executive Summary
This project is a Face Recognition–based Attendance System built with Flask, OpenCV, and the `face_recognition` library (dlib). It enables teachers to capture attendance using the browser camera, recognizes students by comparing face embeddings against a prepared dataset, and records attendance to a database/CSV.

- Core value: Fast, frictionless attendance with minimal hardware (webcam).
- Deployment: Works locally and on servers; clients use a standard web browser.
- Privacy: Processes frames transiently; stores only necessary data (encodings derived from student images).

## 2. Architecture Overview
- Client (Browser):
  - HTML/CSS/JavaScript UI with camera access via `getUserMedia`.
  - Captures a single downscaled frame and sends it to backend as Base64 JPEG.
- Server (Flask `app.py`):
  - API endpoints: authentication, attendance capture (`/api/recognize`), attendance save (`/api/process_attendance`).
  - Face recognition pipeline (OpenCV + face_recognition) using cached encodings.
  - SQLite database via Flask‑SQLAlchemy.
- Storage:
  - SQLite DB (`instance/attendance.db`).
  - Dataset of labeled student images in `dataset/<Student Name>/*.jpg`.
  - Optional cached encodings `encodings.pkl` for fast startup.

## 3. Tech Stack
- Backend: Python, Flask, Flask‑SQLAlchemy
- CV/ML: OpenCV, dlib, face_recognition
- Frontend: HTML, Bootstrap CSS, vanilla JS
- Database: SQLite (can swap to Postgres/MySQL)
- Deployment: Gunicorn (Linux/macOS) or Waitress (Windows) behind Nginx/IIS

## 4. Data Model (DB Schema)
- Teacher(id, username, password_hash)
- Student(id, name, roll_number, class_name)
- Attendance(id, student_id, date, time, status)

CSV exports include columns: name, roll_number, class_name, date, time, status.

## 5. Face Recognition Pipeline
1) Dataset preparation
   - Folders per student name; 5–10 clear images each.
2) Encoding (startup/preload)
   - Images → OpenCV read → RGB → detect faces (HOG; upsample fallbacks) → 128‑D encodings.
   - Cache encodings and names in memory; optionally in `encodings.pkl`.
3) Inference (capture)
   - Client sends downscaled frame.
   - Server detects faces (HOG) and computes encodings.
   - Compare against known encodings using distance; apply confidence threshold (~0.55–0.6).
   - Return recognized names; unknowns filtered.

Why accurate now:
- Aligned dataset preprocessing with runtime detection.
- Multi‑scale/upsample used during building encodings; lean fast path for inference.
- Name equality: folder names must exactly match `Student.name`.

## 6. Request Flow (End‑to‑End)
- Teacher logs in → navigates to Take Attendance.
- Click Start Camera → capture shows live video.
- Click Capture Attendance → client sends Base64 JPEG to `/api/recognize`.
- Server returns list of recognized `students`.
- UI shows recognized names; teacher saves → `/api/process_attendance` persists to DB/CSV.

## 7. Key Files and Responsibilities
- `app.py`: Flask app, routes, recognition API, encoding cache, config.
- `templates/*.html`: Pages (login, dashboard, take attendance UI).
- `static/js/main.js`: UI helpers; camera start/stop; fetch calls.
- `register.py`: CLI to capture dataset images per student via webcam.
- `encode_faces.py`: Optional batch encoder for encodings cache.
- `start.sh`: Convenience script for setup and run on macOS/Linux.
- `instance/attendance.db`: SQLite DB (auto‑created).

## 8. Performance Optimizations
- Client downscales video frame to ~320px width before upload.
- Server uses fast HOG detector with 1 fallback upsample.
- Encodings preloaded and cached (`encodings.pkl`) with dataset mtime invalidation.
- Flask reloader/threading disabled to avoid double‑free crashes and stabilize latency.

Tuning knobs:
- Client width 224–320 tradeoff: speed vs accuracy.
- Confidence threshold 0.55–0.6.
- Skip fallback upsample for speed if faces are close/centered.

## 9. Security and Privacy
- Sessions with server‑side validation for teachers.
- Store only encodings and attendance metadata; no raw webcam frames persisted.
- HTTPS required for camera access on public domains.
- Recommend rotating SECRET_KEY and using secure cookies in production.

## 10. Setup and Installation
- Python 3.10–3.11 recommended (Windows/macOS/Linux).
- `pip install -r requirements.txt`
- On Windows, install prebuilt `dlib` wheel first if needed (see README).
- Initialize DB: `python setup_database.py`
- Run dev: `python app.py`
- Run prod:
  - Linux/macOS: `gunicorn -w 2 -b 0.0.0.0:5000 app:app`
  - Windows: `waitress-serve --port=5000 app:app`

## 11. Deployment
- Reverse proxy with Nginx/IIS; enable HTTPS (Let’s Encrypt on Linux).
- Persist `instance/attendance.db`; back up CSV exports.
- Configure environment: `FLASK_ENV=production`, `SECRET_KEY`, secure cookies.

## 12. Troubleshooting Guide
- Pip install fails on Windows → install `dlib` wheel matching Python; ensure VC++ Redistributable.
- “Address already in use” → stop the other process or change port.
- First capture slow/”UI corrupt” → encodings preloaded; UI button disables with spinner.
- “No face detected” for a student → retake clear images, ensure single face and exact name match.
- macOS crash (double free) → run without reloader/threading; avoid multiple processes.

## 13. API Reference (condensed)
- `POST /api/recognize`
  - Body: `{ image: "data:image/jpeg;base64,..." }`
  - Response: `{ success: true, students: [ { name, confidence } ] }`
- `POST /api/process_attendance`
  - Body: `{ recognized: ["Name1","Name2"], class_name, date }`
  - Response: `{ success: true, saved: N }`

## 14. Demo Script (5–7 minutes)
1) Login with admin/admin123 (mention change later).
2) Show dataset folders; point out exact name matching.
3) Start Camera → Capture → recognized list appears.
4) Save attendance → show row in DB/CSV.
5) Add a new student via `register.py`; capture again.
6) Briefly show code: recognition endpoint and pipeline.

## 15. Likely Judge Questions and Answers
- Q: Why HOG over CNN? A: Faster on CPU; good accuracy with quality images. We can enable CNN if GPU is available.
- Q: How do you handle privacy? A: No raw frames stored; only encodings and attendance metadata. HTTPS enforced.
- Q: Scalability? A: Swap SQLite with Postgres; run behind WSGI with multiple workers; shard encodings per class.
- Q: Spoofing/Photos? A: Basic defense via liveness heuristics (motion/multi-frame) can be added; current scope is classroom convenience.
- Q: Accuracy across lighting/angles? A: Multiple images per student; threshold tuning; fallback detection strategies.

## 16. Roadmap
- Add liveness detection.
- Multi-frame aggregation to boost confidence.
- Batch/stream capture mode.
- Admin UI to manage dataset and encodings.
- Cloud storage and CI/CD deployment templates.

## 17. License and Credits
- Uses `face_recognition` and dlib (BSD-like), OpenCV (Apache 2.0), Flask (BSD).
- Acknowledge upstream libraries.

---
Export to PDF suggestions:
- VS Code/Cursor Markdown PDF extension.
- Pandoc: `pandoc docs/SIH_Attendance_System_Dossier.md -o SIH_Attendance_System_Dossier.pdf`
- Browser print to PDF (open the markdown in a viewer like GitHub).

