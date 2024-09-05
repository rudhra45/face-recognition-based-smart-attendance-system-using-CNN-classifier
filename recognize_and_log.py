import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
import os

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Load the best saved model
model_path = 'data/face_recognition_cnn_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Load label encoder
label_encoder_path = 'data/names.pkl'
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Load face detector
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load background image
background_img_path = 'background.png'
if not os.path.exists(background_img_path):
    raise FileNotFoundError(f"Background image file not found at {background_img_path}")
img_background = cv2.imread(background_img_path)

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize variables
COL_NAMES = ['NAME', 'TIME']
attendance_file_prefix = 'Attendance/Attendance_'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (50, 50))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        class_index = np.argmax(prediction)
        person_name = label_encoder.inverse_transform([class_index])[0]

        # Draw rectangle and label on face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Embed the frame in the background image
    img_background[162:162 + frame.shape[0], 55:55 + frame.shape[1]] = frame

    # Display the frame
    cv2.imshow('Frame', img_background)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [person_name, timestamp]
        attendance_file = attendance_file_prefix + date + ".csv"

        speak(f"Attendance taken for {person_name}.")
        print(f"Attendance taken for {person_name} at {timestamp}.")
        time.sleep(2)

        if os.path.exists(attendance_file):
            with open(attendance_file, "+a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "+a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()