# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from tkinter import messagebox

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for face data collection
skip = 0
face_data = []
dataset_path = "./face_dataset/"

# Prompt the user for their details
user_name = input("Enter your name: ")
admission_number = input("Enter your admission number: ")

# Ensure that the dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Check if the user already exists in the dataset
file_path = os.path.join(dataset_path, f"{admission_number}_{user_name}.npy")
if os.path.exists(file_path):
    messagebox.showinfo("User Exists", "You already exist in the dataset. Access Granted!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Infinite loop for capturing and processing video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check if the frame was successfully captured
    if ret == False:
        continue

    # Detect faces in the graysca
