import cv2
import numpy as np
import pickle
import logging
import threading
import queue
import time
import datetime
import csv
import argparse
import warnings
import os

import torch
import torchvision.ops as tv_ops
from scipy.spatial.distance import cdist

import mediapipe as mp
from ultralytics import YOLO
from keras_facenet import FaceNet
from deep_sort_realtime.deepsort_tracker import DeepSort
import mysql.connector

# ----------------- Patch: Force torchvision.ops.nms to run on CPU -----------------
_orig_nms = tv_ops.nms

def nms_cpu_fallback(boxes, scores, iou_threshold):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)

tv_ops.nms = nms_cpu_fallback

# ----------------- Configuration -----------------
FRAME_DOWNSCALE = 1
INPUT_SIZE = (160, 160)
SIMILARITY_THRESHOLD = 0.5
AUTH_CACHE_TTL = 10
TRACK_MEMORY_TTL = 10
TARGET_FPS = 10
FRAME_INTERVAL = 1 / TARGET_FPS

DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin123',
    'password': 'Petro@123',
    'database': 'RestrictedAreaDB'
}

YOLO_FACE_PATH     = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\model\yolov11n-face.pt"
YOLO_PERSON_PATH   = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\model\yolo11n.pt"
YOLO_OBJECT_PATH   = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\model\inappropriate_behaviour.pt"
ICON_VERIFIED_PATH   = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\icon\verified2.png"
ICON_UNVERIFIED_PATH = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\icon\Unverified.png"
EMBEDDINGS_FILE    = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\scripts\face_encodings.pkl"
TIMESTAMP_FILE     = os.path.join(os.path.dirname(EMBEDDINGS_FILE), "pkltimestamp")
LOG_FILE           = "unauthorized_log.csv"

# Logging & Warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=DeprecationWarning)
cv2.setNumThreads(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Models
face_model   = YOLO(YOLO_FACE_PATH).to(device)
embedder     = FaceNet()
deep_sort     = DeepSort(max_age=10)
person_model = YOLO(YOLO_PERSON_PATH).to(device)
object_model = YOLO(YOLO_OBJECT_PATH).to(device)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
OBJECT_CONFIDENCE_THRESHOLD = 0.25

# Embedding loader and watcher
enbeddings_dict = {}
def load_embeddings():
    global embeddings_dict
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings_dict = pickle.load(f)
    logging.info(f"[Monitor] Reloaded embeddings for {len(embeddings_dict)} identities")

# Initial load
load_embeddings()
# Track last timestamp
last_ts = 0
if os.path.exists(TIMESTAMP_FILE):
    try:
        last_ts = float(open(TIMESTAMP_FILE).read())
    except:
        last_ts = 0

def watch_for_updates(interval=5):
    global last_ts
    while True:
        try:
            ts = float(open(TIMESTAMP_FILE).read())
            if ts > last_ts:
                load_embeddings()
                last_ts = ts
        except FileNotFoundError:
            pass
        time.sleep(interval)

threading.Thread(target=watch_for_updates, daemon=True).start()

# Load and ensure icons have alpha
verified_icon = cv2.imread(ICON_VERIFIED_PATH, cv2.IMREAD_UNCHANGED)
unverified_icon = cv2.imread(ICON_UNVERIFIED_PATH, cv2.IMREAD_UNCHANGED)
def ensure_alpha(icon):
    if icon is None:
        return None
    if icon.ndim == 3 and icon.shape[2] == 3:
        alpha = np.ones((icon.shape[0], icon.shape[1]), dtype=icon.dtype) * 255
        return np.dstack((icon, alpha))
    return icon
verified_icon = ensure_alpha(verified_icon)
unverified_icon = ensure_alpha(unverified_icon)

# Queues & state
frame_queue       = queue.Queue(maxsize=10)
recognition_queue = queue.Queue()
results_queue     = queue.Queue()
stop_threads      = False
authorization_cache = {}
cache_last_refresh  = 0
track_info          = {}

# Helper functions
def connect_to_db():
    return mysql.connector.connect(**DB_CONFIG)

def check_authorization(name):
    global cache_last_refresh
    now = time.time()
    if now - cache_last_refresh > AUTH_CACHE_TTL:
        authorization_cache.clear()
        cache_last_refresh = now
    if name in authorization_cache:
        return authorization_cache[name]
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT Certificate1, Certificate2, Certificate3, Certificate4 FROM IdentityManagement WHERE PersonName=%s",
            (name,)
        )
        row = cur.fetchone()
        conn.close()
    except Exception as e:
        logging.error(f"DB error: {e}")
        return False
    if not row:
        authorization_cache[name] = False
        return False
    cert1, flag2, cert3, flag4 = row
    try:
        cert1 = (datetime.datetime.strptime(cert1, '%Y-%m-%d').date() if isinstance(cert1, str) else cert1)
        cert3 = (datetime.datetime.strptime(cert3, '%Y-%m-%d').date() if isinstance(cert3, str) else cert3)
    except Exception:
        authorization_cache[name] = False
        return False
    valid = (cert1 >= datetime.date.today()) and bool(flag2) and (cert3 >= datetime.date.today()) and bool(flag4)
    authorization_cache[name] = valid
    return valid

def log_unauthorized(name, timestamp, track_id):
    with open(LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([timestamp, name, track_id])

# Utilities
def align_face(image, landmarks):
    left, right = np.array(landmarks[0]), np.array(landmarks[1])
    dY, dX = right[1]-left[1], right[0]-left[0]
    angle = np.degrees(np.arctan2(dY, dX))
    center = ((left[0]+right[0])/2, (left[1]+right[1])/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def preprocess_face(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), INPUT_SIZE)

def extract_embeddings(faces):
    return embedder.embeddings([preprocess_face(f) for f in faces])

def recognize(emb):
    names, embs = zip(*[(n, np.mean(v, axis=0)) for n, v in embeddings_dict.items()])
    dists = cdist([emb], np.vstack(embs), metric='cosine')
    idx = np.argmin(dists)
    sim = 1 - dists[0, idx]
    return (names[idx], sim) if sim >= SIMILARITY_THRESHOLD else (None, sim)

def overlay_text(frame, vcnt, ucnt, fps):
    cv2.putText(frame, f"Authorized: {vcnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Unauthorized: {ucnt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, ts, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

def detect_objects_within_box(img, box):
    x1,y1,x2,y2 = box
    roi = img[y1:y2, x1:x2]
    res = object_model(roi)[0]
    for b in res.boxes:
        if b.conf >= OBJECT_CONFIDENCE_THRESHOLD:
            ox1,oy1,ox2,oy2 = map(int, b.xyxy[0])
            lbl = f"{res.names[int(b.cls)]} ({b.conf.item():.2f})"
            cv2.rectangle(img, (x1+ox1, y1+oy1), (x1+ox2, y1+oy2), (0,0,255), 2)
            cv2.putText(img, lbl, (x1+ox1, y1+oy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

def run_behavior(frame):
    res = person_model(frame)[0]
    for box in res.boxes:
        if int(box.cls)==0:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            detect_objects_within_box(frame,(x1,y1,x2,y2))
    return frame

def recognition_worker():
    while True:
        tid,face_img = recognition_queue.get()
        if face_img is None: break
        emb = extract_embeddings([face_img])[0]
        name,sim = recognize(emb)
        auth = check_authorization(name) if name else False
        results_queue.put((tid, name or "Unknown", auth, sim, time.time()))

def capture_thread(cap):
    last=0
    global stop_threads
    while not stop_threads:
        now=time.time()
        if now-last>=FRAME_INTERVAL:
            ret,fr=cap.read()
            if ret and not frame_queue.full(): frame_queue.put(fr); last=now

def main(source=0):
    global stop_threads, track_info
    threading.Thread(target=recognition_worker, daemon=True).start()
    cap=cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,4096); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
    if not cap.isOpened(): print("Cannot open source, exiting."); return
    threading.Thread(target=capture_thread,args=(cap,),daemon=True).start()
    prev=time.time()
    while True:
        if frame_queue.empty(): time.sleep(0.001); continue
        frame=frame_queue.get()
        frame=run_behavior(frame)
        dets=[]
        for b in face_model(frame)[0].boxes:
            if int(b.cls) == 0:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append(([x1, y1, x2 - x1, y2 - y1], 1.0, 'face'))
        tracks = deep_sort.update_tracks(dets, frame=frame)
        now = time.time()
        # prune old track_info entries
        pruned = {tid: info for tid, info in track_info.items() if now - info['last_seen'] < TRACK_MEMORY_TTL}
        track_info = pruned
        while not results_queue.empty():
            tid, name, auth, sim, tstamp = results_queue.get()
            track_info[tid] = {'name': name, 'auth': auth, 'similarity': sim, 'last_seen': tstamp}

        vcnt = ucnt = 0
        for tr in tracks:
            if not tr.is_confirmed() or tr.time_since_update > 0:
                continue
            tid = tr.track_id
            l, t, w, h = map(int, tr.to_ltwh())
            roi = frame[t:t+h, l:l+w]
            if roi.size == 0:
                continue
            info = track_info.get(tid)
            if not info:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                lm_res = mp_face_mesh.process(rgb)
                if not lm_res.multi_face_landmarks:
                    continue
                pts = lm_res.multi_face_landmarks[0].landmark
                left = (int(pts[33].x * w), int(pts[33].y * h))
                right = (int(pts[263].x * w), int(pts[263].y * h))
                recognition_queue.put((tid, align_face(roi, [left, right])))
                continue
            name, auth, sim = info['name'], info['auth'], info['similarity']
            col = (0, 255, 0) if auth else (0, 0, 255)
            if auth:
                vcnt += 1
            else:
                ucnt += 1
                log_unauthorized(name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tid)

            # Draw bounding box and labels
            cv2.rectangle(frame, (l, t), (l+w, t+h), col, 2)
            text_x = l + w + 5
            cv2.putText(frame, f"{name}", (text_x, t + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            #cv2.putText(frame, f"Sim:{sim:.2f}", (text_x, t + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            
            
                

        fps = 1.0 / (time.time() - prev)
        prev = time.time()
        overlay_text(frame, vcnt, ucnt, fps)
        cv2.imshow("Behaviour Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    recognition_queue.put((None, None))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='rtsp://admin:RMKRCX@169.254.58.237/', help="camera index or video file")
    #parser.add_argument('--video', default=0, help="camera index or video file")
    args = parser.parse_args()
    src = int(args.video) if args.video.isdigit() else args.video
    main(src)
