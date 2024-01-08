# Importing the OpenCV library
import cv2

# Creating a VideoCapture object to capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Infinite loop to continuously capture and display video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # If the frame is not captured successfully, continue to the next iteration
    if ret == False:
        continue

    # Display the video frame in a window titled "video frame"
    cv2.imshow("video frame", frame)

    # Wait for a key event and check if the key 'q' is pressed
    key_pressed = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break out of the loop
    if key_pressed == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
