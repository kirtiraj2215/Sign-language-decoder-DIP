import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

with open("asl_landmark_dataset.pkl", "rb") as f:
    data = pickle.load(f)

dataset = np.array(data[0])
labels = np.array(data[1])

label_counts = Counter(labels)
min_samples_class = [label for label, count in label_counts.items() if count < 2]

dataset = np.array([d for d, l in zip(dataset, labels) if l not in min_samples_class])
labels = np.array([l for l in labels if l not in min_samples_class])

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True, random_state=42)

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

with open("./ASL_landmark_model.pkl", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved successfully.")

def predict_landmarks(landmark_data, model):
    predicted_class = model.predict([landmark_data])
    return predicted_class[0]

example_landmark = dataset[0]
predicted_class = predict_landmarks(example_landmark, model)
print(f"Predicted Class: {predicted_class}")
