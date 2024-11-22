import cv2
import numpy as np
import mediapipe as mp
import pickle

with open("./ASL_landmark_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        return np.array([[lm.x, lm.y] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    landmarks = extract_landmarks(frame)
    if landmarks is not None and len(landmarks) == model.n_features_in_:
        predicted_class = model.predict([landmarks])[0]
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
