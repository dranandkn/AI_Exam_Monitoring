
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available")
    exit()

print("Camera started")
print("Press S to capture an image")
print("Press Q to exit")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture image when S is pressed
    if key == ord('s'):

        cv2.imwrite("captured_image.jpg", frame)

        print("Image saved successfully!")

    # Exit when Q is pressed
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()