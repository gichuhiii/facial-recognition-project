import numpy as np
import cv2
import os
from tkinter import messagebox

# Function to calculate the Euclidean distance between two vectors
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


# Function to perform k-nearest neighbors classification
def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Define the dataset path
dataset_path = "./face_dataset/"

# Initialize lists and dictionaries for face data, labels, and class names
face_data = []
labels = []
class_id = 0
names = {}

# Dataset preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Concatenate face data and labels to create the training set
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

# Create the training set by concatenating face dataset and labels
trainset = np.concatenate((face_dataset, face_labels), axis=1)

# OpenCV font for text rendering
font = cv2.FONT_HERSHEY_SIMPLEX

# Infinite loop for capturing and processing video frames
while True:
    ret, frame = cap.read()

    # If the frame was not successfully captured, continue to the next iteration
    if ret == False:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize flag for visitor detection
    visitor_detected = True

    # Iterate through the detected faces
    for face in faces:
        x, y, w, h = face

        # Get the face region of interest (ROI)
        offset = 5
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Perform k-nearest neighbors classification to recognize the face
        out = knn(trainset, face_section.flatten())

        # Draw the recognized name and a rectangle around the detected face
        cv2.putText(frame, names[int(out)], (x, y - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # If a face is recognized, set the visitor_detected flag to False
        visitor_detected = False

    # Display 'Visitor' if no faces are recognized
    if visitor_detected:
        cv2.putText(frame, 'Visitor', (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the original frame with the detected faces
    cv2.imshow("Faces", frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
