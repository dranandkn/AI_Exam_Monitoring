
import cv2

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access webcam")
    exit()

print("Live camera started!")
print("Press Q to close the camera window.")

while True:

    # Capture one frame
    ret, frame = camera.read()

    # Check whether frame is captured
    if not ret:
        print("Unable to capture frame")
        break

    # Display captured frame
    cv2.imshow("AI Exam Monitoring - Live Camera", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

print("Camera stopped successfully!")