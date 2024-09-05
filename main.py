import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from threading import Thread
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

# Initialize global variables
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def start_capture():
    def capture_faces():
        name = name_entry.get()
        if not name:
            messagebox.showwarning("Input Error", "Please enter your name")
            return
        
        os.makedirs(f'data/faces/{name}', exist_ok=True)
        i = 0
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x + w]
                resized_img = cv2.resize(crop_img, (50, 50))
                cv2.imwrite(f'data/faces/{name}/{i}.jpg', resized_img)
                i += 1
                cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord('q') or i == 100:
                break
        
        video.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Information", f"A new user with the name '{name}' has been added.")
    
    thread = Thread(target=capture_faces)
    thread.start()

def train_model():
    def train():
        # Directories
        dataset_dir = 'data/faces'
        train_dir = dataset_dir
        validation_dir = dataset_dir

        # Image data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=0.2,  # Using 20% of data for validation
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=16,  # Reduced batch size for stability
            class_mode='categorical',
            subset='training'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(128, 128),
            batch_size=16,  # Reduced batch size for stability
            class_mode='categorical',
            subset='validation'
        )

        # Label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(list(train_generator.class_indices.keys()))

        # Model architecture
        model = Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Added dropout to prevent overfitting
            Dense(len(train_generator.class_indices), activation='softmax')
        ])

        # Compile the model with a reduced learning rate
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping and learning rate reduction on plateau
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        # Train the model with callbacks
        model.fit(
            train_generator,
            epochs=50,
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        # Save the model
        os.makedirs('data', exist_ok=True)
        model.save('data/face_recognition_cnn_model.keras')

        # Save the label encoder
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

        messagebox.showinfo("Information", "Model training completed.")

    thread = Thread(target=train)
    thread.start()

def recognize_face():
    def recognize():
        # Load the best saved model
        model_path = 'data/face_recognition_cnn_model.keras'
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found at {model_path}")
            return
        model = load_model(model_path)

        # Load label encoder
        label_encoder_path = 'data/names.pkl'
        if not os.path.exists(label_encoder_path):
            messagebox.showerror("Error", f"Label encoder file not found at {label_encoder_path}")
            return
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Load face detector
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        # Load background image
        background_img_path = 'background.png'
        if not os.path.exists(background_img_path):
            messagebox.showerror("Error", f"Background image file not found at {background_img_path}")
            return
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
                face_img = cv2.resize(face_img, (128, 128))
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

    thread = Thread(target=recognize)
    thread.start()

def speak(str1):
    speak = Dispatch("SAPI.Spvoice")
    speak.Speak(str1)

root = tk.Tk()
root.title("Face Recognition-Based Attendance System")
root.geometry("820x500")

img = tk.PhotoImage(file="user interface background.png")
background_label = tk.Label(root, image=img)
background_label.place(x=1, y=1, relwidth=1, relheight=1)

# Add name label and entry field
name_label = tk.Label(root, text="Enter Name:", font=("Arial", 14), bg="white")
name_label.pack(pady=20)
name_entry = tk.Entry(root, font=("Arial", 14))
name_entry.pack(pady=10)

# Add buttons for capturing faces, training model, and recognizing faces
capture_button = tk.Button(root, text="Capture Faces", font=("Arial", 14), command=start_capture, bg="green", fg="white")
capture_button.pack(pady=10)

train_button = tk.Button(root, text="Train Model", font=("Arial", 14), command=train_model, bg="blue", fg="white")
train_button.pack(pady=10)

recognize_button = tk.Button(root, text="Recognize Face", font=("Arial", 14), command=recognize_face, bg="red", fg="white")
recognize_button.pack(pady=10)

# Run the main loop
root.mainloop()
