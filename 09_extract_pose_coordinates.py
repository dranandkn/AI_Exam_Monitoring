
from ultralytics import YOLO
import cv2

# Load pose model
model = YOLO("yolo11n-pose.pt")

# Open video
cap = cv2.VideoCapture("videos/recorded_video.mp4")

if not cap.isOpened():
    print("Unable to open video")
    exit()

# Read one frame
ret, frame = cap.read()

if not ret:
    print("Unable to read frame")
    cap.release()
    exit()

# Perform pose estimation
results = model.predict(
    source=frame,
    conf=0.35,
    verbose=False
)

# Extract keypoints
for result in results:

    if result.keypoints is None:
        continue

    keypoints = result.keypoints.xy.cpu().numpy()

    print("\nNumber of detected persons:", len(keypoints))

    for person_id, person in enumerate(keypoints):

        print(f"\nPerson {person_id + 1}")

        for keypoint_id, point in enumerate(person):

            x = point[0]
            y = point[1]

            print(
                f"Keypoint {keypoint_id}: "
                f"X = {x:.2f}, Y = {y:.2f}"
            )

cap.release()

print("\nKeypoint extraction completed!")
