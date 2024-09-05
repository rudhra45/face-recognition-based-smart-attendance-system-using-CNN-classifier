import os #allows to interact with the operating system(reading, writing & creating)
import numpy as np #for handling data and for numerical computations and handling arrays
import tensorflow as tf #deep learning library(for building and training the models)
from tensorflow.keras.preprocessing.image import ImageDataGenerator #used for generating batches of tensor image data with real-time data augmentation.(increses the size and diversity)
from tensorflow.keras.models import Sequential #to create linear stack of layers of the model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #for building cnn(convolution-extracting features, max-reduces spatial dimentions, flatten-2d or 3d to 1d vector, dense-process the flattened vector, dropout-prevent overfitting )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import pickle

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
    target_size=(50, 50),
    batch_size=16,  # Reduced batch size for stability
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(50, 50),
    batch_size=16,  # Reduced batch size for stability
    class_mode='categorical',
    subset='validation'
)

# Label encoder
label_encoder = LabelEncoder()
label_encoder.fit(list(train_generator.class_indices.keys()))

# Model architecture
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(50, 50, 3)),
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
