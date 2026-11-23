
import cv2

# Connect to the default webcam
camera = cv2.VideoCapture(0)

# Check camera connection
if not camera.isOpened():
    print("Error: Unable to access webcam")
    exit()

print("Webcam connected successfully!")

# Release camera
camera.release()