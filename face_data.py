import cv2
import numpy as np

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Initialize variables for face data collection
skip = 0
face_data = []
dataset_path = "./face_dataset/"
file_name = input("Enter the name of the person: ")

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
    if key_pressed == ord('q'):
        break

# Convert collected face data to a NumPy array and reshape
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# Save the collected face data as a NumPy array file
np.save(dataset_path + file_name, face_data)
print("Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
