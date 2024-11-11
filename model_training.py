import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

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

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)

print(f"Accuracy Score: {score * 100:.2f}%")

with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model": model}, f)