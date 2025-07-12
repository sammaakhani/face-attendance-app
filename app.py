
import os
import cv2
import numpy as np
import csv
import smtplib
import ssl
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import onnxruntime as ort
from gtts import gTTS
from email.message import EmailMessage
import time
import platform
import urllib.request
import streamlit as st
from PIL import Image
import tempfile
import threading
import pygame

# 1. PATH CONFIGURATION
BASE_DIR = r"E:\sameer makhani\attendance"
CONTENT_DIR = os.path.join(BASE_DIR, 'content')
KNOWN_FACES_DIR = os.path.join(CONTENT_DIR, 'known_faces')
ATTENDANCE_PATH = os.path.join(CONTENT_DIR, 'attendance.csv')
ARCFACE_PATH = os.path.join(CONTENT_DIR, 'arcface.onnx')
PROTO_PATH = os.path.join(BASE_DIR, 'deploy.prototxt')
CAFFEMDL_PATH = os.path.join(BASE_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# 2. VERIFY FILES EXIST
@st.cache_resource
def verify_and_download_models():
    if not os.path.exists(ARCFACE_PATH):
        st.warning("Downloading ArcFace model...")
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

# 3. AUDIO FEEDBACK
pygame.mixer.init()
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)

        # Load and play in Streamlit
        audio_file = open(temp_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

        # Optionally delete the temp file
        audio_file.close()
        os.remove(temp_path)

    except Exception as e:
        st.error(f"Audio Error: {e}")



# 4. EMAIL FUNCTION
def send_email(name, timestamp):
    try:
        msg = EmailMessage()
        msg['Subject'] = f'Attendance Marked: {name}'
        msg['From'] = 'system@institute.com'
        msg['To'] = 'businessoftshirts@gmail.com'
        msg.set_content(f'Student: {name}\nTime: {timestamp}')

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('sam.makhani33@gmail.com', 'hpfi jhnc eixw udzs')
            server.send_message(msg)
    except Exception as e:
        st.error(f"Email Error: {e}")

# 5. LOAD MODELS
@st.cache_resource
def load_models():
    session = ort.InferenceSession(ARCFACE_PATH)
    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, CAFFEMDL_PATH)
    return session, face_net

# 6. GET EMBEDDING
@st.cache_data(show_spinner=False)
def get_embedding(image, _session, _face_net):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104,117,123))
    _face_net.setInput(blob)
    detections = _face_net.forward()

    if detections.shape[2] == 0 or detections[0,0,0,2] < 0.5:
        return None

    h, w = image.shape[:2]
    box = detections[0,0,0,3:7] * np.array([w,h,w,h])
    (x1, y1, x2, y2) = box.astype("int")
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (112,112))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm = (rgb/255.0 - 0.5)/0.5
    inp = np.transpose(norm, (2,0,1))[np.newaxis,:]
    return _session.run(None, {_session.get_inputs()[0].name: inp})[0].flatten()

# 7. LOAD KNOWN FACES
@st.cache_data(show_spinner=True)
def load_known_faces(_session, _face_net):
    known = {}
    names, embeddings = [], []
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, file)
            img = cv2.imread(path)
            emb = get_embedding(img, _session, _face_net)
            if emb is not None:
                name = os.path.splitext(file)[0]
                known[name] = emb
    return known

# 8. STREAMLIT UI
st.title("📸 Face Recognition Attendance")
verify_and_download_models()
session, face_net = load_models()
known_faces = load_known_faces(session, face_net)

captured_image = None
camera_input = st.camera_input("Take a Picture")

if camera_input:
    image = Image.open(camera_input)
    image_array = np.array(image)
    captured_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.image(image, caption="Captured Image", use_column_width=True)

    st.write("🔎 Processing...")
    emb = get_embedding(captured_image, session, face_net)
    if emb is not None:
        match_found = False
        for name, known_emb in known_faces.items():
            similarity = cosine_similarity([emb], [known_emb])[0][0]
            st.write(f"🔍 Match with {name}: {similarity*100:.2f}%")
            if similarity >= 0.5:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(ATTENDANCE_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if os.stat(ATTENDANCE_PATH).st_size == 0:
                        writer.writerow(['Name','Time'])
                    writer.writerow([name, timestamp])
                speak(f"Welcome {name}, attendance marked!")
                send_email(name, timestamp)
                st.success(f"✅ Attendance Marked for {name}")
                match_found = True
                break
        if not match_found:
            speak("No face found try again")
            st.warning("❌ No match found. Try again.")
    else:
        speak("No face detected")
        st.warning("😕 No face detected. Please retake the picture.")
