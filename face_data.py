import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog

# Function to get student details using pop-up windows
def get_student_details():
    student_name = simpledialog.askstring("Input", "Enter student's name:")
    admission_number = simpledialog.askstring("Input", "Enter admission number:")

    return student_name, admission_number

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for face data collection
skip = 0
face_data = []
dataset_path = "./face_dataset/"

# Get student details
student_name, admission_number = get_student_details()

# Ensure that the dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Ensure that the file doesn't already exist before collecting data
file_path = os.path.join(dataset_path, f"{admission_number}_{student_name}.npy")
if os.path.exists(file_path):
    print(f"File '{admission_number}_{student_name}.npy' already exists. Choose a different name.")
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

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # If no faces are detected, continue to the next iteration
    if len(faces) == 0:
        continue

    # Sort the detected faces based on area in descending order
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    # Increment the skip counter
    skip += 1

    # Iterate through the top detected face
    for face in faces[:1]:
        x, y, w, h = face

        # Define an offset to include a margin around the detected face
        offset = 5

        # Extract the face region with the offset
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize the face region to a fixed size (100x100 pixels)
        face_selection = cv2.resize(face_offset, (100, 100))

        # Collect face data every 10 frames
        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        # Display the selected face in a separate window with a dynamic name (str(k))
        cv2.imshow(str(skip), face_selection)

        # Draw a rectangle around the detected face in the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with the detected faces
    cv2.imshow("faces", frame)

    # Check for the 'q' key press to exit the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q') or len(face_data) > 7:
        break

# Convert collected face data to a NumPy array and reshape
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# Save the collected face data as a NumPy array file
np.save(file_path, face_data)
print(f"Dataset saved at: {file_path}")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
