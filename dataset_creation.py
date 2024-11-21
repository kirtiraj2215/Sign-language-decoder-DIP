import os
import cv2
import numpy as np
import pickle

dataset_path = 'asl_dataset'
images = []
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
                image = cv2.resize(image, (64, 64))
                
                images.append(image)
                labels.append(label)

images = np.array(images)
labels = np.array(labels)

with open('asl_dataset.pkl', 'wb') as f:
    pickle.dump((images, labels), f)

print("Dataset has been pickled successfully!")
