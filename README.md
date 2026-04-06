# FaceAttend — Attendance System

Face-recognition attendance system using OpenCV + DeepFace (ArcFace), Flask backend, and a live web dashboard.

## Setup

```bash
# 1. Install dependencies
pip install flask flask-cors deepface opencv-python numpy

# 2. Run the server
cd attendance_system
python app.py
```

Open `static/index.html` in your browser (or serve it via Flask).

---

## Project Structure

```
attendance_system/
├── app.py              ← Flask backend + camera thread + recognition
├── attendance.db       ← SQLite (auto-created)
├── face_db/            ← Enrolled face images
│   └── John Doe/
│       ├── 1.jpg
│       └── 2.jpg
└── static/
    └── index.html      ← Dashboard UI
```

---

## How to Enroll a Face

1. Make sure the person is **in frame** and facing the camera.
2. Type their name in the **Enroll Face** box in the UI.
3. Click **Capture** — it saves the current frame to `face_db/<name>/`.
4. Capture **3–5 photos** per person from slightly different angles for better accuracy.

---

## Switching to Sparsh 5G Camera

In `app.py`, change **one line**:

```python
# Before (built-in webcam)
CAMERA_SOURCE = 0

# After (Sparsh RTSP stream)
CAMERA_SOURCE = "rtsp://admin:admin@<camera_ip>:554/stream1"
```

That's it. Everything else — recognition, logging, UI — stays the same.

---

## API Endpoints

| Endpoint                  | Description                        |
|---------------------------|------------------------------------|
| `GET /video_feed`         | MJPEG stream                       |
| `GET /api/attendance`     | Today's attendance (or ?date=YYYY-MM-DD) |
| `GET /api/attendance/all` | All records                        |
| `GET /api/enrolled`       | List of enrolled people            |
| `GET /api/detections`     | Current frame detections (live)    |
| `GET /api/status`         | Camera + DB status                 |
| `POST /api/enroll`        | Enroll a face `{"name": "..."}`    |

---

## Tips

- Run recognition on every **5th frame** (`FRAME_SKIP = 5`) — adjustable in `app.py`
- Attendance is logged **once per person per day** (deduped by SQLite UNIQUE constraint)
- Delete `face_db/representations_arcface.pkl` if recognition seems stale after re-enrollment
- For better accuracy: enroll in the **same lighting conditions** as deployment