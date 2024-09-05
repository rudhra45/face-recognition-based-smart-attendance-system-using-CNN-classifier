import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the model
model_path = 'data/face_recognition_cnn_model.keras'
model = tf.keras.models.load_model(model_path)

# Set up the data generators
dataset_dir = 'data/faces'
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(50, 50),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for confusion matrix
)

# Get the true labels
Y_true = validation_generator.classes

# Get the predicted labels
Y_pred = model.predict(validation_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Calculate overall metrics
accuracy = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted')
recall = recall_score(Y_true, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

# Print overall metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Classification report (optional, if you still want to print it)
class_report = classification_report(Y_true, Y_pred_classes, target_names=validation_generator.class_indices.keys())
print("\nClassification Report:")
print(class_report)
