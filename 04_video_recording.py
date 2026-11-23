
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available")
    exit()

# Video properties
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 20.0

# Video writer configuration
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(
    "recorded_video.mp4",
    fourcc,
    fps,
    (width, height)
)

print("Video recording started!")
print("Press Q to stop recording.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # Save frame into video
    video.write(frame)

    # Display video
    cv2.imshow("Video Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
video.release()

cv2.destroyAllWindows()

print("Video saved successfully!")