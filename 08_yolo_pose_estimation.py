
from ultralytics import YOLO
import cv2

# Load YOLO11 Pose model
print("Loading YOLO11 Pose model...")

model = YOLO("yolo11n-pose.pt")

# Input video
video_path = "videos/recorded_video.mp4"

# Open recorded video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video")
    exit()

print("Pose estimation started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Perform human pose estimation
    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    # Draw bounding boxes and body keypoints
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow(
        "YOLO11 Human Pose Estimation",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Pose estimation completed!")
