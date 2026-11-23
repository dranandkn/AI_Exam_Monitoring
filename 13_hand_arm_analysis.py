
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

if fps <= 0:
    fps = 20.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "outputs/hand_arm_analysis.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Hand and arm analysis started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track students
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

            points = keypoints[i]

            # Extract left and right arm coordinates
            left_shoulder = points[5]
            right_shoulder = points[6]

            left_elbow = points[7]
            right_elbow = points[8]

            left_wrist = points[9]
            right_wrist = points[10]

            # Check confidence of required keypoints
            if keypoint_conf is not None:

                required_points = [5, 6, 7, 8, 9, 10]

                if any(
                    keypoint_conf[i][p] < 0.35
                    for p in required_points
                ):
                    continue

            # Calculate shoulder centre and width
            shoulder_center_x = (
                left_shoulder[0] + right_shoulder[0]
            ) / 2

            shoulder_width = abs(
                right_shoulder[0] - left_shoulder[0]
            )

            if shoulder_width < 10:
                continue

            # Analyze left arm
            left_raised = (
                left_wrist[1] < left_shoulder[1] - 20
            )

            left_extension = (
                abs(left_wrist[0] - left_shoulder[0])
                / shoulder_width
            )

            # Analyze right arm
            right_raised = (
                right_wrist[1] < right_shoulder[1] - 20
            )

            right_extension = (
                abs(right_wrist[0] - right_shoulder[0])
                / shoulder_width
            )

            observations = []

            if left_raised:
                observations.append("LEFT HAND RAISED")

            if right_raised:
                observations.append("RIGHT HAND RAISED")

            if left_extension > 1.0:
                observations.append("LEFT ARM EXTENDED")

            if right_extension > 1.0:
                observations.append("RIGHT ARM EXTENDED")

            # Display observation
            if observations:

                label = " | ".join(observations)

                color = (0, 165, 255)

            else:

                label = "NO RAISED / EXTENDED ARM OBSERVED"

                color = (0, 255, 0)

            # Get person bounding box position
            x1, y1 = result.boxes.xyxy[i][:2].int().tolist()

            cv2.putText(
                annotated_frame,
                f"ID {track_id}: {label}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2
            )

            # Print observations
            if observations:

                print(
                    f"Student ID {track_id}: "
                    + ", ".join(observations)
                )

    # Display video
    cv2.imshow(
        "Hand and Arm Movement Analysis",
        annotated_frame
    )

    # Save output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Hand and arm analysis completed!")
print("Output saved at:", output_path)
