
from ultralytics import YOLO
import cv2
from collections import deque

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
output_path = "outputs/repeated_head_movement.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# Settings
WINDOW_SECONDS = 15
MOVEMENT_THRESHOLD = 4
SMOOTHING_ALPHA = 0.25
MIN_STATE_DURATION = 0.5

# Store information separately for each tracked person
student_history = {}

frame_number = 0

print("Repeated head movement analysis started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Video timestamp
    video_time = frame_number / fps

    # Detect and track people
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Draw pose and bounding boxes
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

            # Extract nose and shoulder coordinates
            nose = person_points[0]
            left_shoulder = person_points[5]
            right_shoulder = person_points[6]

            # Check keypoint confidence
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

            shoulder_center = (left_x + right_x) / 2

            shoulder_width = abs(right_x - left_x)

            if shoulder_width < 10:
                continue

            # Calculate normalized deviation
            deviation = (
                nose_x - shoulder_center
            ) / shoulder_width

            # Initialize student history
            if track_id not in student_history:

                student_history[track_id] = {
                    "smoothed": deviation,
                    "stable_state": None,
                    "candidate_state": None,
                    "candidate_since": video_time,
                    "changes": deque()
                }

            student = student_history[track_id]

            # Smooth the deviation
            student["smoothed"] = (
                SMOOTHING_ALPHA * deviation
                + (1 - SMOOTHING_ALPHA)
                * student["smoothed"]
            )

            smooth_value = student["smoothed"]

            # Determine current candidate state
            if smooth_value < -0.20:

                current_state = "IMAGE-LEFT"

            elif smooth_value > 0.20:

                current_state = "IMAGE-RIGHT"

            else:

                current_state = "CENTERED"

            # Initialize stable state
            if student["stable_state"] is None:

                student["stable_state"] = current_state
                student["candidate_state"] = current_state
                student["candidate_since"] = video_time

            else:

                stable_state = student["stable_state"]

                # Check whether a new state is appearing
                if current_state != stable_state:

                    if current_state != student["candidate_state"]:

                        student["candidate_state"] = current_state
                        student["candidate_since"] = video_time

                    # Accept the new state only if it persists
                    elif (
                        video_time - student["candidate_since"]
                        >= MIN_STATE_DURATION
                    ):

                        student["stable_state"] = current_state

                        student["changes"].append(video_time)

                else:

                    student["candidate_state"] = current_state
                    student["candidate_since"] = video_time

            # Remove old movement records
            changes = student["changes"]

            while changes and (
                video_time - changes[0] > WINDOW_SECONDS
            ):
                changes.popleft()

            movement_count = len(changes)

            # Generate observation
            if movement_count >= MOVEMENT_THRESHOLD:

                observation = "REPEATED MOVEMENT - REVIEW"

                color = (0, 165, 255)

            else:

                observation = "MOVEMENT OBSERVED"

                color = (0, 255, 0)

            # Display information
            x1, y1 = result.boxes.xyxy[i][:2].int().tolist()

            label = (
                f"ID {track_id}: "
                f"{student['stable_state']}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(25, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Changes: {movement_count} | {observation}",
                (x1, max(50, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Print observation
            if movement_count >= MOVEMENT_THRESHOLD:

                print(
                    f"Student ID {track_id}: "
                    f"{movement_count} head-position changes "
                    f"in the last {WINDOW_SECONDS} seconds"
                )

    # Display video
    cv2.imshow(
        "Repeated Head Movement Analysis",
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

print("Repeated head movement analysis completed!")
print("Output saved at:", output_path)
