import os
import cv2
import numpy as np
import pickle
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = 'asl_dataset'
landmarks = []
labels = []

label_map = {chr(i): i - 97 for i in range(97, 123)}
label_map.update({str(i): i + 26 for i in range(10)})

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        label = label_map.get(folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image)
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    landmark_coords = []
                    for lm in hand_landmarks.landmark:
                        landmark_coords.append([lm.x, lm.y])
                    landmarks.append(np.array(landmark_coords).flatten())
                    labels.append(label)

landmarks = np.array(landmarks)
labels = np.array(labels)

with open('asl_landmark_dataset.pkl', 'wb') as f:
    pickle.dump((landmarks, labels), f)
