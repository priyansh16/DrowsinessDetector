#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import time

class DrowsinessDetector:
    def __init__(self, model_path='../models/best_model.keras', alarm_path = '../Alert.wav', drowsy_time=0.2):
        #load trained model and alarm 
        self.model = load_model(model_path)
        self.alarm_path = alarm_path
        self.drowsy_time = drowsy_time
        self.alarm_duration = 3.0  # 2 seconds alarm duration
        self.alarm_start_time = None  # Track when alarm started
        # Initialize cascade clasifier
        self.face_cacade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def alarm(self):
        if not mixer.get_init():  # initialize if not already initialized
            mixer.init()
            mixer.music.load(self.alarm_path)
        if not mixer.music.get_busy():  # play sould if not already playing
            mixer.music.play()
            self.alarm_start_time = time.time()
        
    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)
        sleep_start = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("issue capturing the image, try again")
                break
            
            # Check if alarm should be stopped
            if self.alarm_start_time and time.time() - self.alarm_start_time >= self.alarm_duration:
                if mixer.get_init() and mixer.music.get_busy():
                    mixer.music.stop()
                self.alarm_start_time = None
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cacade.detectMultiScale(gray_frame, 1.3, 5)
            
            for(x, y, w, h) in faces:
                roi_gray = gray_frame[y: y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2: # detected both the eyes
                    eyes_images = []
                    for (ex, ey, ew, eh) in eyes[:2]:
                        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                        eye_roi = cv2.resize(eye_roi, (24,24)) # Resizing to match model input
                        eye_roi = eye_roi/255.0 # Normalization
                        eyes_images.append(eye_roi)
                    
                    #input for the model 
                    eyes_array = np.array(eyes_images).reshape(-1, 24, 24, 1)
                    prediction = self.model.predict(eyes_array)
                    # pred_values = prediction.flatten()
                    
                    # Enhanced debugging information
                    # print(f"Eye predictions: {pred_values}")
                    
                    # Consider eyes closed if BOTH eyes are mostly closed
                    # eyes_closed = all(pred < 0.5 for pred in pred_values)
                    # state = "CLOSED" if eyes_closed else "OPEN"
                    # print(f"Current eye state: {state}")
                    
                    # Display eye state and predictions on frame
                    # cv2.putText(frame, f"Eye State: {state}", 
                            #   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    #if closed eyes detected
                    if prediction.mean() < 0.5:  # Changed condition
                        if sleep_start is None:
                            sleep_start = time.time()
                        elif time.time() - sleep_start > self.drowsy_time:
                            self.alarm()
                            # Draw red rectangle and warning text
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)  # Red rectangle
                            cv2.putText(frame, 'DROWSINESS DETECTED!', (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        sleep_start = None
                        # Draw blue rectangle for normal state
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            cv2.imshow('Drowsiness Detection', frame)
            
            # Check for 'q' key press with a 1ms wait
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        # Stop the alarm if it's playing when we quit
        if mixer.get_init():
            mixer.music.stop()
            mixer.quit()
            