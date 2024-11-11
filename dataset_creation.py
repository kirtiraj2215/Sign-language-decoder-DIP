import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

data_dir = './asl_dataset'
dataset = []
labels = []

def is_image_file(file_name):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_name.lower().endswith(ext) for ext in image_extensions)

for directory in os.listdir(data_dir):
    path = os.path.join(data_dir, directory)
    
    if not os.path.isdir(path):
        continue

    for img_path in os.listdir(path):
        if not is_image_file(img_path):
            continue

        normalized_landmarks = []
        x_coordinates, y_coordinates = [], []

        image_path = os.path.join(path, img_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to load image at path: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = hands.process(image_rgb)
        hand_landmarks = processed_image.multi_hand_landmarks

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                landmark_coordinates = hand_landmark.landmark

                for coordinates in landmark_coordinates:
                    x_coordinates.append(coordinates.x)
                    y_coordinates.append(coordinates.y)

                min_x, min_y = min(x_coordinates), min(y_coordinates)

                for coordinates in landmark_coordinates:
                    normalized_x = coordinates.x - min_x
                    normalized_y = coordinates.y - min_y
                    normalized_landmarks.extend((normalized_x, normalized_y))

            dataset.append(normalized_landmarks)
            labels.append(directory)

with open("./ASL.pickle", "wb") as f:
    pickle.dump({"dataset": dataset, "labels": labels}, f)

print("Dataset creation complete and saved as ASL.pickle")
