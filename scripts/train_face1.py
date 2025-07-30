import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
import mediapipe as mp
import datetime

# Paths to data and output file

data_dir = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\face_data"
output_file = r"C:\Users\muham\OneDrive\Desktop\inappropriate_behaviour_v2_\scripts\face_encodings.pkl"
# Timestamp flag file for monitoring bridge
timestamp_file = os.path.join(os.path.dirname(output_file), "pkltimestamp")

# Ensure training directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Face detection helper (MediaPipe)
mp_face_detection = mp.solutions.face_detection

def scale_box(box, scale_factor=1.2):
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    w, h = int(w * scale_factor), int(h * scale_factor)
    x, y = cx - w // 2, cy - h // 2
    return max(0, x), max(0, y), w, h


def detect_faces_mediapipe(image, face_detection):
    """Detect and crop faces using MediaPipe."""
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            x, y, w, h = scale_box((x, y, w, h))
            face = image[y:y + h, x:x + w]
            faces.append((face, (x, y, w, h)))
    return faces

# Preprocessing for FaceNet
INPUT_SIZE = (160, 160)

def preprocess_image(image):
    """Resize and convert image for FaceNet."""
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), INPUT_SIZE)

# Initialize embedder
embedder = FaceNet()

def extract_embeddings(faces):
    """Generate embeddings for list of face crops."""
    imgs = [preprocess_image(face) for face, _ in faces]
    return embedder.embeddings(imgs)

# Training function

def train_model():
    """Scan data directory, extract embeddings, and pickle them."""
    embeddings_dict = {}
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        for person in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person)
            if not os.path.isdir(person_path):
                continue
            person_embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                faces = detect_faces_mediapipe(image, face_detection)
                if faces:
                    embeddings = extract_embeddings(faces)
                    person_embeddings.extend(embeddings)
            if person_embeddings:
                embeddings_dict[person] = np.array(person_embeddings)
    # Save to disk
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_dict, f)
    # Update timestamp flag for monitoring
    with open(timestamp_file, "w") as flag:
        flag.write(str(datetime.datetime.now().timestamp()))

# Entry point for training
if __name__ == "__main__":
    train_model()
