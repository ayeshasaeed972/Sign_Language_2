import cv2
import numpy as np
import pyttsx3
import time
import vlc
from tensorflow.keras.models import load_model

model = load_model("sign_model.h5")
categories = np.load("categories.npy")

img_size = 64
cap = cv2.VideoCapture(0)
player = vlc.MediaPlayer()

# Initialize text-to-speech engine and helper variables
engine = pyttsx3.init()
last_label = None
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size)) / 255.0
    reshaped = resized.reshape(1, img_size, img_size, 1)


    result = model.predict(reshaped)
    confidence = np.max(result) * 100
    label = categories[np.argmax(result)]

    # Speak if confident and not repeated
    if label != last_label and confidence > 90 and (time.time() - last_time > 2):
        engine.say(f"The detected letter is {label}")
        engine.runAndWait()
        last_label = label
        last_time = time.time()

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {label} ({confidence:.2f}%)", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()