import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe's hands module for hand detection and landmark estimation
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Define the main dataset directory
data_dir = './asl_dataset'
dataset = []
labels = []

# Function to check if a file is an image based on extension
def is_image_file(file_name):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_name.lower().endswith(ext) for ext in image_extensions)

# Loop through each directory representing digits and letters in the dataset folder
for directory in os.listdir(data_dir):
    path = os.path.join(data_dir, directory)
    
    # Check if the path is a directory (ignore files that may be in the main folder)
    if not os.path.isdir(path):
        continue

    # Loop through each image file in the current class directory
    for img_path in os.listdir(path):
        if not is_image_file(img_path):  # Skip non-image files
            continue

        normalized_landmarks = []  # List to store normalized x, y coordinates
        x_coordinates, y_coordinates = [], []  # Temporary lists for x and y coordinates

        # Read the image
        image_path = os.path.join(path, img_path)
        image = cv2.imread(image_path)

        # Check if the image was successfully loaded
        if image is None:
            print(f"Warning: Unable to load image at path: {image_path}")
            continue  # Skip to the next image if loading failed

        # Convert the image from BGR to RGB format (required by MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands using MediaPipe's hand processing method
        processed_image = hands.process(image_rgb)

        # Get the hand landmarks (if any hand is detected in the image)
        hand_landmarks = processed_image.multi_hand_landmarks

        if hand_landmarks:  # If hand landmarks are found
            for hand_landmark in hand_landmarks:
                landmark_coordinates = hand_landmark.landmark  # Get individual landmark coordinates

                # Extract the x and y coordinates of all landmarks
                for coordinates in landmark_coordinates:
                    x_coordinates.append(coordinates.x)
                    y_coordinates.append(coordinates.y)

                # Find the minimum x and y values to normalize the coordinates
                min_x, min_y = min(x_coordinates), min(y_coordinates)

                # Normalize the landmarks by subtracting the minimum x and y values
                for coordinates in landmark_coordinates:
                    normalized_x = coordinates.x - min_x
                    normalized_y = coordinates.y - min_y
                    normalized_landmarks.extend((normalized_x, normalized_y))  # Add normalized values to the list

            # Append the normalized landmarks to the dataset
            dataset.append(normalized_landmarks)

            # Append the label (class name, such as '0', '1', 'A', 'B', etc.) for the current directory
            labels.append(directory)

# Save the dataset and labels using pickle
with open("./ASL.pickle", "wb") as f:
    pickle.dump({"dataset": dataset, "labels": labels}, f)

print("Dataset creation complete and saved as ASL.pickle")