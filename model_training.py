import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

label_counts = Counter(dataset["labels"])
print(f"Class distribution: {label_counts}")

min_samples_class = [label for label, count in label_counts.items() if count < 2]
print(f"Classes with fewer than 2 samples: {min_samples_class}")

dataset["dataset"] = [d for d, l in zip(dataset["dataset"], dataset["labels"]) if l not in min_samples_class]
dataset["labels"] = [l for l in dataset["labels"] if l not in min_samples_class]

label_counts = Counter(dataset["labels"])
print(f"Updated class distribution: {label_counts}")

data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

# Encode string labels to integers
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, shuffle=True, random_state=42)

# Ensure all classes are present in the training set
for class_label in range(len(label_encoder.classes_)):
    if class_label not in np.unique(y_train):
        X_train = np.append(X_train, [np.zeros(X_train.shape[1])], axis=0)
        y_train = np.append(y_train, [class_label])
        
# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)

print(f"Accuracy Score: {score * 100:.2f}%")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()

#Saving the trained model in ASL_model.p
with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model": model, "label_encoder": label_encoder}, f)
