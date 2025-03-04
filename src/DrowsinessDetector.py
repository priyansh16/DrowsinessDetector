#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import time

class DrowsinessDetector:
    def __init__(self, model_path='../models/best_model.keras', alarm_path = '../Alert.wav', drowsy_time=2.0):
        #load trained model and alarm 
        self.model = load_model(model_path)
        self.alarm_path = alarm_path
        self.drowsy_time = drowsy_time
        self.alarm_playing = False  # Tracks if alarm is playing
        self.sleep_start = None # Tracks when eyes are first closed
        
        # Initialize cascade clasifier
        self.face_cacade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        #initialize pygame mixer
        mixer.init()
        self.alarm_channel = mixer.Sound(self.alarm_path)
        
    def play_alarm(self):
        """Plays alarm sound if not already playing"""
        if not self.alarm_playing:
            self.alarm_channel.play(-1)  # playing sound until manually stopped
            self.alarm_playing = True
    
    def stop_alarm(self):
        """Stop alarm sound if it's already playing"""
        if self.alarm_playing:
            self.alarm_channel.stop()
            self.alarm_playing = False
            
            
    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("issue capturing the image, try again")
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cacade.detectMultiScale(gray_frame, 1.3, 5)
            
            eyes_detected = False # flag to track if eyes are detected
            
            for(x, y, w, h) in faces:
                roi_gray = gray_frame[y: y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2: # detected both the eyes
                    eyes_detected = True
                    eyes_images = []
                    
                    for (ex, ey, ew, eh) in eyes[:2]:
                        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                        eye_roi = cv2.resize(eye_roi, (24,24)) # To match model input
                        eye_roi = eye_roi/255.0 # Normalization
                        eyes_images.append(eye_roi)
                    
                    #Preparing input for the model 
                    eyes_array = np.array(eyes_images).reshape(-1, 24, 24, 1)
                    prediction = self.model.predict(eyes_array)
                    
                    if prediction.mean() < 0.5:  #if closed eyes are detected
                        if self.sleep_start is None:
                            self.sleep_start = time.time()
                        elif time.time() - self.sleep_start >= self.drowsy_time:
                            self.play_alarm()
                            # Red rectangle and warning in real time
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)  
                            cv2.putText(frame, 'DROWSINESS DETECTED!', (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.sleep_start = None
                        self.stop_alarm()
                        # Blue rectangle if normal state in real time
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            #If eyes are not detected, resetting the timer and stopping alarm.            
            if not eyes_detected:
                self.sleep_start = None
                self.stop_alarm()
                    
            cv2.imshow('Drowsiness Detection', frame)
            
            # 'q' for stopping the alarm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm() #Stopping alarm while quitting detection.
        
            