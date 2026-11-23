
from ultralytics import YOLO
import cv2

# Load YOLO11 Pose model
print("Loading YOLO11 Pose model...")

model = YOLO("yolo11n-pose.pt")

# Input video
video_path = "videos/recorded_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "outputs/head_position_analysis.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps if fps > 0 else 20.0,
    (width, height)
)

# Store smoothed deviation for each tracked person
head_history = {}

# Smoothing factor
alpha = 0.25

print("Head-position analysis started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track people
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Draw pose skeleton and bounding boxes
    annotated_frame = result.plot()

    if (
        result.boxes is not None
        and result.boxes.id is not None
        and result.keypoints is not None
    ):

        track_ids = result.boxes.id.int().cpu().tolist()

        keypoints = result.keypoints.xy.cpu().numpy()

        keypoint_conf = result.keypoints.conf

        if keypoint_conf is not None:
            keypoint_conf = keypoint_conf.cpu().numpy()

        for i, track_id in enumerate(track_ids):

            if i >= len(keypoints):
                continue

            person_points = keypoints[i]

            # Nose, left shoulder and right shoulder
            nose = person_points[0]
            left_shoulder = person_points[5]
            right_shoulder = person_points[6]

            # Check whether important keypoints are reliable
            if keypoint_conf is not None:

                if (
                    keypoint_conf[i][0] < 0.35
                    or keypoint_conf[i][5] < 0.35
                    or keypoint_conf[i][6] < 0.35
                ):
                    continue

            nose_x = float(nose[0])
            left_x = float(left_shoulder[0])
            right_x = float(right_shoulder[0])

            # Calculate shoulder centre
            shoulder_center = (left_x + right_x) / 2

            # Calculate shoulder width
            shoulder_width = abs(right_x - left_x)

            if shoulder_width < 10:
                continue

            # Calculate normalized deviation
            deviation = (
                nose_x - shoulder_center
            ) / shoulder_width

            # Smooth the deviation
            if track_id not in head_history:

                head_history[track_id] = deviation

            else:

                head_history[track_id] = (
                    alpha * deviation
                    + (1 - alpha) * head_history[track_id]
                )

            smooth_deviation = head_history[track_id]

            # Determine head-position observation
            if smooth_deviation < -0.20:

                observation = "IMAGE-LEFT"

            elif smooth_deviation > 0.20:

                observation = "IMAGE-RIGHT"

            else:

                observation = "CENTERED"

            # Display observation on video
            label = f"ID {track_id}: {observation}"

            x1, y1 = result.boxes.xyxy[i][:2].int().tolist()

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display processed video
    cv2.imshow(
        "Head Position Analysis",
        annotated_frame
    )

    # Save processed video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Head-position analysis completed!")
print("Output saved at:", output_path)
