import cv2
import numpy as np
import os
import tensorflow as tf

model = tf.keras.models.load_model('i3d_convlstm_accident_detection_model.h5')

model.summary()

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

def predict_accident(video_path):
    video_data = preprocess_video(video_path)
    
    video_data = np.expand_dims(video_data, axis=0)
    prediction = model.predict(video_data)
    
    if np.argmax(prediction) == 1:
        print("Accident Detected!")
    else:
        print("No Accident Detected.")


predict_accident('video data.mp4')