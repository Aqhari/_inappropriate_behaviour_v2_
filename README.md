# Behavior Monitoring System

A real-time face recognition and behavior monitoring solution built with YOLO, FaceNet, MediaPipe, and DeepSort. It automates embedding training, live monitoring, and integration into a CI/CD pipeline.

---

## 📂 Project Structure

```
your-project/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions pipeline (Python 3.12)
├── face_data/                 # Per-person image folders
├── model_checkpoints/         # YOLO weight files
├── scripts/
│   ├── train_face1.py         # Embedding trainer
│   ├── monitoring2.py         # Live monitoring script
│   ├── auto_trainer.py        # Watchdog → retrain glue
│   ├── face_encodings.pkl     # Generated at runtime (git-ignored)
│   └── pkltimestamp           # Generated at runtime (git-ignored)
├── requirements.txt           # Python dependencies
└── .gitignore                 # Ignore generated artifacts
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/behavior-monitor.git
cd behavior-monitor
```

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\Scripts\\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Data & Models

* **face\_data/**: Populate with subfolders named after each person, each containing their face images.
* **model\_checkpoints/**: Ensure YOLO weight files (`.pt`) are placed here:

  * `yolov11n-face.pt` (face detector)
  * `yolo11n.pt` (person detector)
  * `inappropriate_behaviour.pt` (object/behavior detector)

---

## ⚙️ Usage

### 1. Start Auto-Trainer

Automatically retrains embeddings when `face_data/` changes:

```bash
python scripts/auto_trainer.py \
  --face_data scripts/face_data \
  --train_script scripts/train_face1.py
```

### 2. Run Live Monitor

Launch the real-time monitoring and recognition:

```bash
python scripts/monitoring2.py --video <camera_source>
```

* Replace `<camera_source>` with your camera index or RTSP URL.
* Press `q` to quit the live window.

---

## 📈 Continuous Integration

GitHub Actions is configured in `.github/workflows/ci.yml` to:

1. **Lint** code with Black & Flake8
2. **Smoke-test** the auto-trainer script
3. **Build & Publish** a Docker image on `main` branch merges

---

## 📝 .gitignore

```gitignore
# Python
venv/
__pycache__/

# Generated artifacts
scripts/face_encodings.pkl
scripts/pkltimestamp

# Model files (if you want to download separately)
model_checkpoints/
```

---

## 🎓 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
