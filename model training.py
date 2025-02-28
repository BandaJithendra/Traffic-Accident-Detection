import cv2
import numpy as np
import os
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPooling3D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

VIDEO_DATASET_DIR = r"D:\GPCET\Final Project\Project\Dataset"

def preprocess_video(video_path, img_size=(224, 224), sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, img_size)
        normalized_frame = resized_frame / 255.0 
        frames.append(normalized_frame)

    cap.release()

    while len(frames) < sequence_length:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))

    return np.array(frames)

def process_video_dataset(dataset_dir, sequence_length=30, img_size=(224, 224)):
    video_data = []
    labels = []

    for label in ['accident', 'no_accident']:
        label_dir = os.path.join(dataset_dir, label)
        
        if not os.path.exists(label_dir):
            print(f"Error: Directory '{label_dir}' does not exist! Skipping...")
            continue  

        video_files = [f for f in os.listdir(label_dir) if f.endswith(('.mp4', '.avi'))]

        if len(video_files) == 0:
            print(f"Warning: No videos found in '{label_dir}'!")
            continue  

        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            print(f"Processing video: {video_file} ({label})")
            frames = preprocess_video(video_path, img_size=img_size, sequence_length=sequence_length)
            video_data.append(frames)
            labels.append(1 if label == 'accident' else 0)

    return np.array(video_data), np.array(labels)


def build_i3d_convlstm_model(input_shape=(30, 224, 224, 3)):
    model = Sequential()

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model




X, y = process_video_dataset(VIDEO_DATASET_DIR)

if len(X) == 0 or len(y) == 0:
    raise ValueError("Error: No data found! Check dataset path and files.")


model = build_i3d_convlstm_model()

model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

model.save('i3d_convlstm_accident_detection_model.h5')

print("Model training completed and saved successfully!")
