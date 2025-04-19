import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initializing MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load model
model = load_model('asl_model.keras')

# Prediction smoothing
pred_history = []
SMOOTHING_WINDOW = 5  # Number of frames to average

def get_hand_roi(image, landmarks):
    x_coords = [landmark.x * image.shape[1] for landmark in landmarks.landmark]
    y_coords = [landmark.y * image.shape[0] for landmark in landmarks.landmark]
    
    padding = 30
    x_min, x_max = int(min(x_coords)) - padding, int(max(x_coords)) + padding
    y_min, y_max = int(min(y_coords)) - padding, int(max(y_coords)) + padding
    
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
    
    return image[y_min:y_max, x_min:x_max]

def predict_sign(roi):
    img = cv2.resize(roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)[0]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_roi = get_hand_roi(frame, hand_landmarks)
            if hand_roi.size > 0:
                prediction = predict_sign(hand_roi)
                pred_history.append(np.argmax(prediction))
                
                if len(pred_history) > SMOOTHING_WINDOW:
                    pred_history.pop(0)
                
                if pred_history:
                    predicted_class = max(set(pred_history), key=pred_history.count)
                    confidence = np.max(prediction)
                    predicted_letter = chr(65 + predicted_class)
                    
                    # Display
                    cv2.putText(frame, f"Sign: {predicted_letter} ({confidence:.2f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('ASL Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()