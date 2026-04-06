import cv2
import sqlite3
import os
import time
import threading
import base64
import json
from datetime import datetime, date
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 0                        # swap to "rtsp://..." for Sparsh
# CAMERA_SOURCE = "rtsp://admin:admin123@192.168.128.10:554/stream1"
FACE_DB_PATH  = "./face_db"
DB_PATH       = "./attendance.db"
FRAME_SKIP    = 5                        # run recognition every N frames
# ────────────────────────────────────────────────────────────────────────────

# Global state
latest_frame       = None
latest_detections  = []        # list of {name, confidence, bbox}
frame_lock         = threading.Lock()
detection_lock     = threading.Lock()
camera_control_lock = threading.Lock()
session_lock       = threading.Lock()
cap                = None
camera_running     = False
camera_thread_handle = None
session_unique_students = set()
session_started_at = None

# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            date      TEXT NOT NULL,
            time      TEXT NOT NULL,
            confidence REAL,
            UNIQUE(name, date)          -- one entry per person per day
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS enrolled_faces (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT UNIQUE NOT NULL,
            added TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_attendance(name, confidence):
    today     = date.today().isoformat()
    now_time  = datetime.now().strftime("%H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR IGNORE INTO attendance (name, date, time, confidence) VALUES (?,?,?,?)",
            (name, today, now_time, round(confidence, 3))
        )
        conn.commit()
        inserted = c.rowcount > 0
    except Exception as e:
        print(f"DB error: {e}")
        inserted = False
    finally:
        conn.close()
    return inserted

def get_attendance(filter_date=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if filter_date:
        c.execute("SELECT name, date, time, confidence FROM attendance WHERE date=? ORDER BY time DESC", (filter_date,))
    else:
        c.execute("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC")
    rows = [{"name": r[0], "date": r[1], "time": r[2], "confidence": r[3]} for r in c.fetchall()]
    conn.close()
    return rows

def get_enrolled():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, added FROM enrolled_faces ORDER BY name")
    rows = [{"name": r[0], "added": r[1]} for r in c.fetchall()]
    conn.close()
    return rows

def register_enrolled(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO enrolled_faces (name, added) VALUES (?,?)",
              (name, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def start_camera():
    global camera_running, camera_thread_handle, session_unique_students, session_started_at
    with camera_control_lock:
        if camera_running and camera_thread_handle and camera_thread_handle.is_alive():
            return False
        camera_running = True
        with session_lock:
            session_unique_students = set()
            session_started_at = datetime.now().isoformat()
        camera_thread_handle = threading.Thread(target=camera_thread, daemon=True)
        camera_thread_handle.start()
    return True


def stop_camera():
    global camera_running, cap, latest_frame, latest_detections, camera_thread_handle
    thread_to_join = None
    with camera_control_lock:
        was_running = camera_running
        camera_running = False
        if camera_thread_handle and camera_thread_handle.is_alive():
            thread_to_join = camera_thread_handle
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None

    if thread_to_join:
        thread_to_join.join(timeout=2.0)

    with camera_control_lock:
        if camera_thread_handle and not camera_thread_handle.is_alive():
            camera_thread_handle = None

    with frame_lock:
        latest_frame = None
    with detection_lock:
        latest_detections = []

    return was_running

# ── Camera + Recognition thread ──────────────────────────────────────────────
def camera_thread():
    global latest_frame, latest_detections, camera_running, cap

    # Lazy-import DeepFace so Flask starts fast
    from deepface import DeepFace

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Camera open failed. Check CAMERA_SOURCE.")
        with frame_lock:
            latest_frame = None
        with detection_lock:
            latest_detections = []
        camera_running = False
        cap.release()
        cap = None
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_count = 0

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1

        # Draw existing detections on every frame (cheap)
        annotated = frame.copy()
        with detection_lock:
            current_detections = list(latest_detections)

        for det in current_detections:
            x, y, w, h = det.get("bbox", (0, 0, 0, 0))
            name = det.get("name", "Unknown")
            conf = det.get("confidence", 0)
            color = (0, 220, 100) if name != "Unknown" else (0, 80, 220)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({conf:.0%})" if name != "Unknown" else "Unknown"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with frame_lock:
            latest_frame = annotated.copy()

        # Run recognition every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            run_recognition(frame, DeepFace)

    if cap is not None:
        cap.release()
        cap = None
    camera_running = False


def run_recognition(frame, DeepFace):
    global latest_detections

    # First detect faces (fast)
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception:
        return

    new_detections = []
    db_files = os.listdir(FACE_DB_PATH) if os.path.exists(FACE_DB_PATH) else []
    has_db = any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in db_files) or \
             any(os.path.isdir(os.path.join(FACE_DB_PATH, d)) for d in db_files)

    for face_obj in face_objs:
        region = face_obj.get("facial_area", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        conf_score = face_obj.get("confidence", 0)

        if conf_score < 0.5 or w < 30:
            continue

        name = "Unknown"
        match_conf = 0.0

        if has_db:
            try:
                results = DeepFace.find(
                    img_path=frame,
                    db_path=FACE_DB_PATH,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )
                if results and len(results[0]) > 0:
                    top = results[0].iloc[0]
                    identity_path = top.get("identity", "")
                    # Extract name from path: face_db/PersonName/img.jpg → PersonName
                    parts = identity_path.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        name = parts[-2]
                    else:
                        name = os.path.splitext(parts[-1])[0]
                    dist = top.get("distance", 1.0)
                    match_conf = max(0, 1 - dist)

                    if match_conf > 0.4:
                        logged = log_attendance(name, match_conf)
                        if logged:
                            print(f"✅ Logged attendance: {name} ({match_conf:.1%})")
                    else:
                        name = "Unknown"
            except Exception as e:
                pass  # no match or DB error

        new_detections.append({
            "name": name,
            "confidence": match_conf,
            "bbox": (x, y, w, h)
        })

        if name != "Unknown":
            with session_lock:
                session_unique_students.add(name)

    with detection_lock:
        latest_detections = new_detections


# ── MJPEG stream ─────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30fps cap


# ── API Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return app.send_static_file("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/attendance")
def api_attendance():
    filter_date = request.args.get("date", date.today().isoformat())
    return jsonify(get_attendance(filter_date))

@app.route("/api/attendance/all")
def api_attendance_all():
    return jsonify(get_attendance())

@app.route("/api/enrolled")
def api_enrolled():
    return jsonify(get_enrolled())

@app.route("/api/detections")
def api_detections():
    with detection_lock:
        dets = [{"name": d["name"], "confidence": d["confidence"]} for d in latest_detections]
    return jsonify(dets)

@app.route("/api/status")
def api_status():
    with session_lock:
        session_count = len(session_unique_students)
        started_at = session_started_at
    with detection_lock:
        current_total_faces = len(latest_detections)
        current_known_faces = len([d for d in latest_detections if d.get("name") != "Unknown"])

    return jsonify({
        "camera": camera_running,
        "face_db_path": FACE_DB_PATH,
        "enrolled_count": len(get_enrolled()),
        "today_count": len(get_attendance(date.today().isoformat())),
        "session_count": session_count,
        "session_started_at": started_at,
        "current_total_faces": current_total_faces,
        "current_known_faces": current_known_faces
    })


@app.route("/api/surveillance/start", methods=["POST"])
def api_surveillance_start():
    changed = start_camera()
    return jsonify({
        "success": True,
        "changed": changed,
        "camera": camera_running,
        "message": "Surveillance started" if changed else "Surveillance already running"
    })


@app.route("/api/surveillance/stop", methods=["POST"])
def api_surveillance_stop():
    changed = stop_camera()
    return jsonify({
        "success": True,
        "changed": changed,
        "camera": camera_running,
        "message": "Feed stopped" if changed else "Feed already stopped"
    })

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """
    Capture current RTSP/server frame and save for a given name.
    POST body: { "name": "John Doe" }
    """
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    with frame_lock:
        frame = latest_frame

    if frame is None:
        return jsonify({"error": "No frame available from camera"}), 503

    person_dir = os.path.join(FACE_DB_PATH, name)
    os.makedirs(person_dir, exist_ok=True)
    img_count = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    img_path = os.path.join(person_dir, f"{img_count + 1}.jpg")
    cv2.imwrite(img_path, frame)
    register_enrolled(name)

    pkl_path = os.path.join(FACE_DB_PATH, "representations_arcface.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)

    return jsonify({"success": True, "message": f"Saved image for '{name}' ({img_count + 1} total)"})


@app.route("/api/enroll_image", methods=["POST"])
def api_enroll_image():
    """
    Accept a base64-encoded image from the browser's own webcam.
    POST body: { "name": "John Doe", "image": "data:image/jpeg;base64,..." }
    """
    data = request.json or {}
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        import base64 as b64mod
        img_bytes = b64mod.b64decode(image_b64)
        np_arr   = np.frombuffer(img_bytes, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Image decode error: {e}"}), 400

    person_dir = os.path.join(FACE_DB_PATH, name)
    os.makedirs(person_dir, exist_ok=True)
    img_count = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    img_path  = os.path.join(person_dir, f"{img_count + 1}.jpg")
    cv2.imwrite(img_path, frame)
    register_enrolled(name)

    # Invalidate DeepFace cache so new face is picked up immediately
    pkl_path = os.path.join(FACE_DB_PATH, "representations_arcface.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)

    total = img_count + 1
    return jsonify({"success": True, "saved": total,
                    "message": f"Photo {total} saved for '{name}'"})


if __name__ == "__main__":
    init_db()
    os.makedirs(FACE_DB_PATH, exist_ok=True)
    start_camera()
    print("🚀 Attendance server running at http://localhost:5050")
    print(f"📁 Face DB: {os.path.abspath(FACE_DB_PATH)}")
    print(f"🗄️  SQLite:  {os.path.abspath(DB_PATH)}")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False, threaded=True)