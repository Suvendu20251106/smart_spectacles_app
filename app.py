import streamlit as st
import cv2
import numpy as np
import pyttsx3
from gtts import gTTS
import os
import tempfile
import playsound
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # lightweight model

# Text-to-Speech engine
engine = pyttsx3.init()

def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)
    playsound.playsound(temp_path)
    os.remove(temp_path)
    engine.say(text)
    engine.runAndWait()

st.title("Smart Spectacles for the Visually Impaired")

st.write("This app simulates how smart glasses can detect obstacles using AI and provide audio guidance.")

option = st.radio("Select Input Type", ["Webcam", "Upload Image"])

if option == "Webcam":
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        FRAME_WINDOW.image(annotated_frame, channels='BGR')

        detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]
        if 'person' in detected_classes or 'car' in detected_classes:
            speak("Obstacle ahead! Be careful!")

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)[0]
        annotated_frame = results.plot()

        st.image(annotated_frame, caption="Processed Image", channels='BGR')

        detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]
        if 'person' in detected_classes or 'car' in detected_classes:
            speak("Obstacle ahead! Be careful!")
