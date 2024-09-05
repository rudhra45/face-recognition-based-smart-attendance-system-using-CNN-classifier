# Automated-face-recogniztion-based-smart-attendance-sytem-using-CNN-classifier
->Introduction
  This project aims to develop an intelligent face detection and recognition system for smart attendance logging. The system   uses machine learning techniques and computer vision to detect and recognize faces, providing a robust solution for         automating attendance tracking.

->Technologies Used
    Python: Programming language.
    OpenCV: Library for computer vision tasks.
    TensorFlow/Keras: Framework for building and training the CNN model.
    Haar Cascades: Pre-trained model for face detection.
    Pickle: For saving and loading the label encoder.

->Project Workflow
  1.Add User Faces:
    Captures images of new users to be recognized.
    Saves the captured images into the data/faces/ directory.

  2.Train Model:
    Trains the Convolutional Neural Network (CNN) model using the captured face images.
    Saves the trained model in data/face_recognition_cnn_model.keras and the label encoder in data/names.pkl.
  
  3.Recognize and Log Attendance:
    Recognizes faces in real-time using the trained model.
    Logs attendance of recognized users in a file or database.


->Order of Execution
  1)Install the required dependencies by running:
    use the command: pip install -r requirements.txt

  2)Run add_faces.py to capture and save images of new users.
    use the command: python add_faces.py

  3)After adding user faces, train the model by running train_model.py.
    use the command: python train_model.py

  4)Run recognize_and_log.py to start the face recognition system and log attendance.
    use the command: python recognize_and_log.py

  5)For evaluating the model and generating the confusion matrix for the model.
    use the command: python confusion_matrix.py

->Features
  Face Detection: Utilizes the Haar Cascade classifier for real-time face detection.
  Face Recognition: Implements a CNN model to accurately recognize faces.
  Attendance Logging: Logs the attendance of recognized users into a log file or database.
  GUI Interface: Provides a user-friendly interface for administrators to manage the system.
