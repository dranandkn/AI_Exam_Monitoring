
from ultralytics import YOLO
import cv2

# Load YOLO11 pretrained object detection model
print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

# Input video
video_path = "videos/recorded_video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
output_path = "outputs/yolo_detected_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps if fps > 0 else 20.0,
    (width, height)
)

print("Video detection started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Perform YOLO detection
    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Save processed frame
    out.write(annotated_frame)

    # Display processed frame
    cv2.imshow(
        "YOLO11 - Video Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video detection completed!")
print("Output saved at:", output_path)
