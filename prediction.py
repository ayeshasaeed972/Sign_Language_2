import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model # type: ignore

# ===== Configuration =====
MODEL_PATH = 'best_model_100epochs.keras'  # Your trained model
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]
ROI_SIZE = 300          # Detection area size
MIN_CONFIDENCE = 0.85   # Only accept predictions above this threshold
SAVE_CAPTURES = True    # Save high-confidence frames

# ===== Performance Optimization =====
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# ===== Load Model =====
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ===== Preprocessing Function =====
def preprocess_frame(frame):
    """Process frames identically to training data"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

# ===== Initialize Webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

# Create captures directory if needed
if SAVE_CAPTURES:
    os.makedirs("captures", exist_ok=True)

# ===== Main Prediction Loop =====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror and flip frame
        frame = cv2.flip(frame, 1)
        
        # Calculate ROI (centered)
        height, width = frame.shape[:2]
        roi_x = (width - ROI_SIZE) // 2
        roi_y = (height - ROI_SIZE) // 2
        roi = frame[roi_y:roi_y+ROI_SIZE, roi_x:roi_x+ROI_SIZE]
        
        # Preprocess and predict
        input_tensor = preprocess_frame(roi)
        predictions = model.predict(input_tensor, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_class = CLASSES[predicted_idx]
        
        # Only accept high-confidence predictions
        display_class = predicted_class if confidence > MIN_CONFIDENCE else "?"
        
        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 10:
            fps = int(fps_frame_count / (time.time() - fps_start_time))
            fps_start_time = time.time()
            fps_frame_count = 0
        
        # Visualization
        # 1. ROI Rectangle
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x+ROI_SIZE, roi_y+ROI_SIZE), 
                     (0, 255, 0), 2)
        
        # 2. Prediction Text
        cv2.putText(frame, f"Sign: {display_class}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps}", 
                   (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2)
        
        # 3. Confidence Bar
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (50, 120), 
                     (50 + bar_width, 140), 
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (50, 120), 
                     (50 + 300, 140), 
                     (255, 255, 255), 2)
        
        # Save high-confidence frames
        if SAVE_CAPTURES and confidence > 0.95:
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"captures/{predicted_class}_{timestamp}.jpg", roi)
        
        # Display
        cv2.imshow("Sign Language Recognition", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera resources released")