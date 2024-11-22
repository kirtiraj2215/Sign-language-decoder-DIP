import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from collections import Counter

with open("/Users/kirtirajjamnotiya/Desktop/Semester 5/DS601/Project/asl_dataset.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data[0].shape)

dataset = np.array(data[0])
labels = np.array(data[1])

label_counts = Counter(labels)
print(f"Class distribution: {label_counts}")

min_samples_class = [label for label, count in label_counts.items() if count < 2]
print(f"Classes with fewer than 2 samples: {min_samples_class}")

dataset = np.array([d for d, l in zip(dataset, labels) if l not in min_samples_class])
labels = np.array([l for l in labels if l not in min_samples_class])

label_counts = Counter(labels)
print(f"Updated class distribution: {label_counts}")

dataset_flattened = dataset.reshape(dataset.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(dataset_flattened, labels, test_size=0.2, shuffle=True, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

all_classes = sorted(np.unique(labels))
cm = confusion_matrix(y_test, y_pred, labels=all_classes)

custom_labels = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=custom_labels, yticklabels=custom_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved successfully.")

def predict_image(image_path, model):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return None
    
    processed_image = np.random.random(len(dataset_flattened[0]))
    
    predicted_class = model.predict([processed_image])
    return predicted_class[0]

image_path = ""
predicted_class = predict_image(image_path, model)
print(f"Predicted Class: {predicted_class}")
