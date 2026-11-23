
from ultralytics import YOLO
import cv2
import csv
import os

# -----------------------------------------
# 1. LOAD YOLO11 POSE MODEL
# -----------------------------------------

print("Loading YOLO11 Pose model...")

model = YOLO("yolo11n-pose.pt")

# -----------------------------------------
# 2. VIDEO CONFIGURATION
# -----------------------------------------

video_path = "videos/recorded_video.mp4"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Unable to open examination video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 20.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "outputs/head_orientation_improved.mp4"
csv_path = "outputs/head_orientation_report.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Unable to create output video.")
    cap.release()
    exit()

# -----------------------------------------
# 3. DETECTION PARAMETERS
# -----------------------------------------

CONFIDENCE_THRESHOLD = 0.35

LEFT_THRESHOLD = -0.20
RIGHT_THRESHOLD = 0.20

SMOOTHING_ALPHA = 0.25

MIN_STATE_DURATION = 0.5

MIN_EYE_DISTANCE = 6

# -----------------------------------------
# 4. STUDENT TRACKING HISTORY
# -----------------------------------------

student_history = {}

frame_number = 0

# -----------------------------------------
# 5. CSV REPORT CONFIGURATION
# -----------------------------------------

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

fieldnames = [
    "Timestamp",
    "Student_ID",
    "Nose_Deviation",
    "Raw_Head_Position",
    "Stable_Head_Position",
    "Direction_Changes"
]

writer = csv.DictWriter(
    csv_file,
    fieldnames=fieldnames
)

writer.writeheader()

print("Starting head orientation analysis...")

# -----------------------------------------
# 6. PROCESS VIDEO FRAME BY FRAME
# -----------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    video_time = frame_number / fps

    # YOLO11 Pose + ByteTrack
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    # -------------------------------------
    # 7. EXTRACT STUDENT KEYPOINTS
    # -------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
        and result.keypoints is not None
    ):

        track_ids = (
            result.boxes.id.int().cpu().tolist()
        )

        boxes = result.boxes.xyxy.cpu().numpy()

        keypoints = result.keypoints.xy.cpu().numpy()

        keypoint_conf = result.keypoints.conf

        if keypoint_conf is not None:
            keypoint_conf = keypoint_conf.cpu().numpy()

        for i, track_id in enumerate(track_ids):

            points = keypoints[i]

            nose = points[0]
            left_eye = points[1]
            right_eye = points[2]

            # ---------------------------------
            # 8. CHECK FACIAL KEYPOINT QUALITY
            # ---------------------------------

            if keypoint_conf is not None:

                required_points = [0, 1, 2]

                if any(
                    keypoint_conf[i][p]
                    < CONFIDENCE_THRESHOLD
                    for p in required_points
                ):
                    continue

            # ---------------------------------
            # 9. CALCULATE EYE CENTRE
            # ---------------------------------

            eye_center_x = (
                left_eye[0] + right_eye[0]
            ) / 2

            eye_distance = abs(
                left_eye[0] - right_eye[0]
            )

            if eye_distance < MIN_EYE_DISTANCE:
                continue

            # ---------------------------------
            # 10. CALCULATE NOSE DEVIATION
            # ---------------------------------

            deviation = (
                float(nose[0]) - eye_center_x
            ) / eye_distance

            # ---------------------------------
            # 11. CLASSIFY HEAD ORIENTATION
            # ---------------------------------

            if deviation < LEFT_THRESHOLD:

                current_state = "IMAGE-LEFT"

            elif deviation > RIGHT_THRESHOLD:

                current_state = "IMAGE-RIGHT"

            else:

                current_state = "FRONT/UNCERTAIN"

            # ---------------------------------
            # 12. INITIALIZE STUDENT HISTORY
            # ---------------------------------

            if track_id not in student_history:

                student_history[track_id] = {
                    "smoothed": deviation,
                    "stable_state": current_state,
                    "candidate_state": current_state,
                    "candidate_since": video_time,
                    "direction_changes": 0
                }

            student = student_history[track_id]

            # ---------------------------------
            # 13. SMOOTH HEAD ORIENTATION
            # ---------------------------------

            student["smoothed"] = (
                SMOOTHING_ALPHA * deviation
                + (1 - SMOOTHING_ALPHA)
                * student["smoothed"]
            )

            smooth_value = student["smoothed"]

            if smooth_value < LEFT_THRESHOLD:

                current_state = "IMAGE-LEFT"

            elif smooth_value > RIGHT_THRESHOLD:

                current_state = "IMAGE-RIGHT"

            else:

                current_state = "FRONT/UNCERTAIN"

            # ---------------------------------
            # 14. STABILIZE HEAD POSITION
            # ---------------------------------

            if current_state != student["stable_state"]:

                if current_state != student["candidate_state"]:

                    student["candidate_state"] = current_state

                    student["candidate_since"] = video_time

                elif (
                    video_time - student["candidate_since"]
                    >= MIN_STATE_DURATION
                ):

                    student["stable_state"] = current_state

                    student["direction_changes"] += 1

            else:

                student["candidate_state"] = current_state

                student["candidate_since"] = video_time

            stable_state = student["stable_state"]

            direction_changes = student["direction_changes"]

            # ---------------------------------
            # 15. DISPLAY FACIAL LANDMARKS
            # ---------------------------------

            nose_x = int(nose[0])
            nose_y = int(nose[1])

            eye_x = int(eye_center_x)
            eye_y = int(
                (left_eye[1] + right_eye[1]) / 2
            )

            cv2.circle(
                annotated_frame,
                (nose_x, nose_y),
                5,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                annotated_frame,
                (eye_x, eye_y),
                5,
                (255, 0, 0),
                -1
            )

            cv2.line(
                annotated_frame,
                (nose_x, nose_y),
                (eye_x, eye_y),
                (0, 255, 255),
                2
            )

            # ---------------------------------
            # 16. DISPLAY STUDENT INFORMATION
            # ---------------------------------

            x1, y1, x2, y2 = boxes[i]

            x1 = int(x1)
            y1 = int(y1)

            cv2.putText(
                annotated_frame,
                f"STUDENT ID: {track_id}",
                (x1, max(20, y1 - 65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"HEAD: {stable_state}",
                (x1, max(40, y1 - 45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"DEVIATION: {smooth_value:.2f}",
                (x1, max(60, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"CHANGES: {direction_changes}",
                (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # ---------------------------------
            # 17. SAVE STUDENT DATA TO CSV
            # ---------------------------------

            writer.writerow({
                "Timestamp": round(video_time, 2),
                "Student_ID": track_id,
                "Nose_Deviation": round(smooth_value, 3),
                "Raw_Head_Position": current_state,
                "Stable_Head_Position": stable_state,
                "Direction_Changes": direction_changes
            })

    # -------------------------------------
    # 18. DISPLAY AND SAVE VIDEO
    # -------------------------------------

    cv2.putText(
        annotated_frame,
        "IMPROVED HEAD ORIENTATION ANALYSIS",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Head Orientation Analysis",
        annotated_frame
    )

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------------------
# 19. RELEASE RESOURCES
# -----------------------------------------

cap.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()

print("\nHEAD ORIENTATION ANALYSIS COMPLETED")

print("Total frames processed:", frame_number)

print("Output video:", output_path)

print("CSV report:", csv_path)
