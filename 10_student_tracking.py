
from ultralytics import YOLO
import cv2

# Load YOLO11 Pose model
print("Loading YOLO11 Pose model...")

model = YOLO("yolo11n-pose.pt")

# Input video
video_path = "videos/recorded_video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "outputs/student_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps if fps > 0 else 20.0,
    (width, height)
)

print("Student tracking started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection and ByteTrack tracking
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes, pose and tracking IDs
    annotated_frame = result.plot()

    # Extract tracking IDs
    if result.boxes is not None and result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        for track_id in track_ids:
            print("Tracked Person ID:", track_id)

    # Save output
    out.write(annotated_frame)

    # Display output
    cv2.imshow(
        "Student Tracking - ByteTrack",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Student tracking completed!")
print("Output saved at:", output_path)
