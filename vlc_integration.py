import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model # type: ignore
import time
from collections import deque

# ===== VLC Configuration =====
VLC_PASSWORD = "admin"  # Set this in VLC preferences
VLC_HOST = "http://localhost:8080"

# Mapping of signs to VLC commands
SIGN_TO_COMMAND = {
    '1': 'pl_play',      # Play
    '2': 'pl_pause',     # Pause
    '3': 'pl_stop',      # Stop
    '4': 'pl_next',      # Next track
    '5': 'pl_previous',  # Previous track
    'a': 'volume&val=+10',  # Volume up
    'b': 'volume&val=-10'   # Volume down
}

# ===== Model Configuration =====
MODEL_PATH = 'best_model_100epochs.keras'  # Your trained model
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]

# ===== Gesture Detection Settings =====
MIN_CONFIDENCE = 0.85    # Only accept predictions with 85%+ confidence
GESTURE_HISTORY_LENGTH = 5  # Require 5 consistent predictions
ROI_SIZE = 300           # Size of detection area (square)

# Initialize gesture history
gesture_history = deque(maxlen=GESTURE_HISTORY_LENGTH)

# ===== Initialize VLC Control =====
def vlc_command(command):
    """Send command to VLC HTTP interface"""
    try:
        response = requests.get(
            f"{VLC_HOST}/requests/status.xml?command={command}",
            auth=("", VLC_PASSWORD))
        return response.status_code == 200
    except Exception as e:
        print(f"VLC Control Error: {e}")
        return False

# ===== Load Model =====
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ===== Initialize Webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cv2.namedWindow("Sign Language VLC Control", cv2.WINDOW_NORMAL)

# ===== Preprocessing Function =====
def preprocess_frame(frame):
    """Process frames identically to training data"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

# ===== Main Loop =====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for more intuitive control
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Define Region of Interest (ROI)
        roi_x = (width - ROI_SIZE) // 2
        roi_y = (height - ROI_SIZE) // 2
        roi = frame[roi_y:roi_y+ROI_SIZE, roi_x:roi_x+ROI_SIZE]
        
        # Preprocess and predict
        input_tensor = preprocess_frame(roi)
        predictions = model.predict(input_tensor, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_sign = CLASSES[predicted_idx]
        
        # Add to gesture history
        if confidence > MIN_CONFIDENCE:
            gesture_history.append(predicted_sign)
        else:
            gesture_history.append(None)
        
        # Check for consistent gesture
        current_command = None
        if len(gesture_history) == GESTURE_HISTORY_LENGTH:
            # Only act if all recent predictions agree
            if all(g == gesture_history[0] for g in gesture_history):
                current_command = SIGN_TO_COMMAND.get(gesture_history[0])
        
        # Execute VLC command
        if current_command:
            success = vlc_command(current_command)
            if success:
                print(f"Executed: {current_command}")
                # Clear history after successful command
                gesture_history.clear()
        
        # Display UI
        # 1. ROI Rectangle
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x+ROI_SIZE, roi_y+ROI_SIZE), 
                     (0, 255, 0), 2)
        
        # 2. Prediction Info
        cv2.putText(frame, f"Sign: {predicted_sign}", 
                   (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                   (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
        
        # 3. Command Status
        if current_command:
            cmd_name = list(SIGN_TO_COMMAND.keys())[list(SIGN_TO_COMMAND.values()).index(current_command)]
            cv2.putText(frame, f"VLC: {cmd_name.upper()}", 
                       (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow("Sign Language VLC Control", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")