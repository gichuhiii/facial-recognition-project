import cv2
import numpy as np

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # Iterate through the detected faces
    for face in faces[:1]:
        # Extract the face coordinates and dimensions
        x, y, w, h = face

        # Define an offset to include a margin around the detected face
        offset = 10

        # Extract the face region with the offset
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize the face region to a fixed size (100x100 pixels)
        face_selection = cv2.resize(face_offset, (100, 100))

        # Display the selected face in a separate window named "Face"
        cv2.imshow("Face", face_selection)

        # Draw a rectangle around the detected face in the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with the detected faces
    cv2.imshow("faces", frame)

    # Check for the 'q' key press to exit the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
