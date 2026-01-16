import os
import cv2
import numpy as np
import csv
import smtplib
import urllib.request
import tempfile
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from email.message import EmailMessage
from PIL import Image
from gtts import gTTS
import onnxruntime as ort

# ==============================
# 1. STREAMLIT SECRETS (EMAIL)
# ==============================
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASS = st.secrets["EMAIL_PASS"]

# ==============================
# 2. PATH CONFIGURATION (CLOUD SAFE)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(BASE_DIR, "content")
KNOWN_FACES_DIR = os.path.join(CONTENT_DIR, "known_faces")
ATTENDANCE_PATH = os.path.join(CONTENT_DIR, "attendance.csv")
ARCFACE_PATH = os.path.join(CONTENT_DIR, "arcface.onnx")
PROTO_PATH = os.path.join(CONTENT_DIR, "deploy.prototxt")
CAFFEMDL_PATH = os.path.join(CONTENT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)

# ==============================
# 3. DOWNLOAD MODELS (IF MISSING)
# ==============================
@st.cache_resource
def download_models():
    if not os.path.exists(ARCFACE_PATH):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/ailia-models/arcface/arcfaceresnet100-8.onnx",
            ARCFACE_PATH
        )

    if not os.path.exists(PROTO_PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            PROTO_PATH
        )

    if not os.path.exists(CAFFEMDL_PATH):
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            CAFFEMDL_PATH
        )

# ==============================
# 4. LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    session = ort.InferenceSession(ARCFACE_PATH)
    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, CAFFEMDL_PATH)
    return session, face_net

# ==============================
# 5. AUDIO FEEDBACK
# ==============================
def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            audio_bytes = open(f.name, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
        os.remove(f.name)
    except:
        pass

# ==============================
# 6. EMAIL FUNCTION
# ==============================
def send_email(name, timestamp):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Attendance Marked: {name}"
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_USER
        msg.set_content(f"Student: {name}\nTime: {timestamp}")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    except:
        pass

# ==============================
# 7. FACE EMBEDDING
# ==============================
def get_embedding(image, session, face_net):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104, 117, 123)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    if detections.shape[2] == 0 or detections[0, 0, 0, 2] < 0.5:
        return None

    h, w = image.shape[:2]
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype(int)

    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (112, 112))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

    norm = (rgb / 255.0 - 0.5) / 0.5
    inp = np.transpose(norm, (2, 0, 1))[None, :]

    return session.run(None, {session.get_inputs()[0].name: inp})[0].flatten()

# ==============================
# 8. LOAD KNOWN FACES
# ==============================
@st.cache_data
def load_known_faces(session, face_net):
    known = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
            emb = get_embedding(img, session, face_net)
            if emb is not None:
                known[os.path.splitext(file)[0]] = emb
    return known

# ==============================
# 9. STREAMLIT UI
# ==============================
st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("📸 Face Recognition Attendance System")

download_models()
session, face_net = load_models()
known_faces = load_known_faces(session, face_net)

camera_input = st.camera_input("Take a picture")

if camera_input:
    image = Image.open(camera_input)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Captured Image", use_column_width=True)

    emb = get_embedding(image_np, session, face_net)

    if emb is None:
        speak("No face detected")
        st.warning("No face detected")
    else:
        matched = False
        for name, known_emb in known_faces.items():
            score = cosine_similarity([emb], [known_emb])[0][0]

            if score > 0.55:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(ATTENDANCE_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    if os.stat(ATTENDANCE_PATH).st_size == 0:
                        writer.writerow(["Name", "Time"])
                    writer.writerow([name, timestamp])

                speak(f"Welcome {name}")
                send_email(name, timestamp)
                st.success(f"Attendance marked for {name}")
                matched = True
                break

        if not matched:
            speak("No match found")
            st.error("No match found")
