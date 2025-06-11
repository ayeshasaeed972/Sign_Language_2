import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import requests
from collections import deque

class SignLanguageVLCController:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language VLC Controller")
        
        # Model configuration
        self.model = load_model('best_model_100epochs.keras')
        self.CLASSES = ['0','1','2','3','4','5','6','7','8','9',
                       'a','b','c','d','e','f','g','h','i','j',
                       'k','l','m','n','o','p','q','r','s','t',
                       'u','v','w','x','y','z']
        
        # VLC configuration
        self.VLC_PASSWORD = "admin"
        self.VLC_HOST = "http://localhost:8080"
        
        # Command mapping
        self.SIGN_TO_COMMAND = {
            '1': ('Play', 'pl_play'),
            '2': ('Pause', 'pl_pause'),
            '3': ('Stop', 'pl_stop'),
            '4': ('Next', 'pl_next'),
            '5': ('Previous', 'pl_previous'),
            'a': ('Vol +', 'volume&val=+10'),
            'b': ('Vol -', 'volume&val=-10')
        }
        
        # Initialize UI
        self.setup_ui()
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.gesture_history = deque(maxlen=5)
        self.min_confidence = 0.85
        self.update()

    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack()
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="VLC Controls", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        buttons = [
            ("Play", 'pl_play'),
            ("Pause", 'pl_pause'),
            ("Stop", 'pl_stop'),
            ("Next", 'pl_next'),
            ("Previous", 'pl_previous'),
            ("Vol +", 'volume&val=+10'),
            ("Vol -", 'volume&val=-10')
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            ttk.Button(control_frame, text=text, 
                      command=lambda c=cmd: self.vlc_command(c)).grid(
                          row=i//4, column=i%4, padx=5, pady=2)
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Sign Detection", padding="10")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sign_label = ttk.Label(status_frame, text="Detected Sign: None")
        self.sign_label.pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(status_frame, text="Confidence: 0%")
        self.confidence_label.pack(anchor=tk.W)
        
        self.command_label = ttk.Label(status_frame, text="Last Command: None")
        self.command_label.pack(anchor=tk.W)
        
        # Settings panel
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Min Confidence:").pack(side=tk.LEFT)
        self.confidence_slider = ttk.Scale(settings_frame, from_=0.5, to=1.0, value=0.85)
        self.confidence_slider.pack(side=tk.LEFT, padx=5)
        self.confidence_slider.bind("<Motion>", self.update_confidence)
        
        ttk.Button(settings_frame, text="Exit", command=self.cleanup).pack(side=tk.RIGHT)

    def update_confidence(self, event):
        """Update minimum confidence threshold"""
        self.min_confidence = self.confidence_slider.get()

    def vlc_command(self, command):
        """Send command to VLC"""
        try:
            requests.get(
                f"{self.VLC_HOST}/requests/status.xml?command={command}",
                auth=("", self.VLC_PASSWORD))
            cmd_name = next((k[0] for k,v in self.SIGN_TO_COMMAND.items() if v[1] == command), "Unknown")
            self.command_label.config(text=f"Last Command: {cmd_name}")
        except Exception as e:
            print(f"VLC Error: {e}")

    def update(self):
        """Update the video feed and predictions"""
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Define ROI
            roi_size = 300
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Draw ROI
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x+roi_size, roi_y+roi_size), 
                         (0, 255, 0), 2)
            
            # Preprocess and predict
            processed = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(processed, (64, 64))
            processed = processed / 255.0
            input_tensor = np.expand_dims(processed, axis=(0, -1))
            
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_sign = self.CLASSES[predicted_idx]
            
            # Update UI
            self.sign_label.config(text=f"Detected Sign: {predicted_sign}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Gesture recognition
            if confidence > self.min_confidence:
                self.gesture_history.append(predicted_sign)
            else:
                self.gesture_history.append(None)
            
            # Execute command if consistent gesture
            if len(self.gesture_history) == 5 and all(g == self.gesture_history[0] for g in self.gesture_history):
                if predicted_sign in self.SIGN_TO_COMMAND:
                    _, command = self.SIGN_TO_COMMAND[predicted_sign]
                    self.vlc_command(command)
            
            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        
        self.root.after(10, self.update)

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageVLCController(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
