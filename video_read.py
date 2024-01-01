# -*- coding: utf-8 -*-
" The following code checks if the webcam is running "

# Import the OpenCV library
import cv2

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Infinite loop for capturing and displaying video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if ret == False:
        continue

    # Display the captured frame in a window named "video frame"
    cv2.imshow("video frame", frame)

    # Check for the 'q' key press to exit the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
