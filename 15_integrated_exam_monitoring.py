
from ultralytics import YOLO
import cv2
import csv
from collections import deque

# ==========================================
# 1. LOAD AI MODELS
# ==========================================

print("Loading YOLO11 Pose model...")
pose_model = YOLO("yolo11n-pose.pt")

print("Loading YOLO11 Object Detection model...")
object_model = YOLO("yolo11n.pt")

# ==========================================
# 2. VIDEO INPUT AND OUTPUT
# ==========================================

video_path = "videos/recorded_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open examination video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 20.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "outputs/integrated_monitoring.mp4"
csv_path = "outputs/monitoring_report.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# ==========================================
# 3. MONITORING PARAMETERS
# ==========================================

WINDOW_SECONDS = 15
MOVEMENT_THRESHOLD = 4
SMOOTHING_ALPHA = 0.25
MIN_STATE_DURATION = 0.5

PHONE_CLASS = 67

# Store history separately for each tracked person
student_history = {}

frame_number = 0
total_phone_detections = 0

# ==========================================
# 4. CREATE CSV REPORT
# ==========================================

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

fieldnames = [
    "Timestamp",
    "Student_ID",
    "Head_Position",
    "Head_Movement_Count",
    "Arm_Observation",
    "Phone_Detected",
    "Monitoring_Status"
]

writer = csv.DictWriter(
    csv_file,
    fieldnames=fieldnames
)

writer.writeheader()

print("\nIntegrated monitoring started...")
print("Press Q to stop processing.")

# ==========================================
# 5. PROCESS VIDEO FRAME BY FRAME
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    video_time = frame_number / fps

    # --------------------------------------
    # A. PERSON DETECTION AND POSE TRACKING
    # --------------------------------------

    pose_results = pose_model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    pose_result = pose_results[0]

    # Draw pose skeletons and bounding boxes
    annotated_frame = pose_result.plot()

    # --------------------------------------
    # B. MOBILE PHONE DETECTION
    # --------------------------------------

    phone_results = object_model.predict(
        source=frame,
        classes=[PHONE_CLASS],
        conf=0.35,
        verbose=False
    )

    phone_result = phone_results[0]

    phones = []

    if phone_result.boxes is not None:

        for box in phone_result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].int().cpu().tolist()
            )

            confidence = float(box.conf[0])

            phones.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })

    total_phone_detections += len(phones)

    # --------------------------------------
    # C. GET TRACKING IDs AND KEYPOINTS
    # --------------------------------------

    if (
        pose_result.boxes is not None
        and pose_result.keypoints is not None
        and pose_result.boxes.id is not None
    ):

        track_ids = (
            pose_result.boxes.id.int().cpu().tolist()
        )

        boxes = pose_result.boxes.xyxy.cpu().numpy()

        keypoints = (
            pose_result.keypoints.xy.cpu().numpy()
        )

        keypoint_conf = pose_result.keypoints.conf

        if keypoint_conf is not None:
            keypoint_conf = keypoint_conf.cpu().numpy()

        # ----------------------------------
        # D. ANALYZE EACH TRACKED STUDENT
        # ----------------------------------

        for i, track_id in enumerate(track_ids):

            points = keypoints[i]

            if keypoint_conf is not None:

                required_points = [0, 5, 6, 9, 10]

                if any(
                    keypoint_conf[i][p] < 0.35
                    for p in required_points
                ):
                    continue

            # Extract body coordinates
            nose = points[0]

            left_shoulder = points[5]
            right_shoulder = points[6]

            left_wrist = points[9]
            right_wrist = points[10]

            shoulder_center = (
                left_shoulder[0] + right_shoulder[0]
            ) / 2

            shoulder_width = abs(
                right_shoulder[0] - left_shoulder[0]
            )

            if shoulder_width < 10:
                continue

            # ----------------------------------
            # E. HEAD POSITION ANALYSIS
            # ----------------------------------

            deviation = (
                float(nose[0]) - shoulder_center
            ) / shoulder_width

            if track_id not in student_history:

                student_history[track_id] = {
                    "smoothed": deviation,
                    "stable_state": None,
                    "candidate_state": None,
                    "candidate_since": video_time,
                    "changes": deque()
                }

            student = student_history[track_id]

            student["smoothed"] = (
                SMOOTHING_ALPHA * deviation
                + (1 - SMOOTHING_ALPHA)
                * student["smoothed"]
            )

            smooth_value = student["smoothed"]

            if smooth_value < -0.20:
                current_state = "IMAGE-LEFT"

            elif smooth_value > 0.20:
                current_state = "IMAGE-RIGHT"

            else:
                current_state = "CENTERED"

            # Confirm a new state after a minimum duration
            if student["stable_state"] is None:

                student["stable_state"] = current_state
                student["candidate_state"] = current_state
                student["candidate_since"] = video_time

            else:

                if current_state != student["stable_state"]:

                    if current_state != student["candidate_state"]:

                        student["candidate_state"] = current_state
                        student["candidate_since"] = video_time

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
            while (
                student["changes"]
                and video_time - student["changes"][0]
                > WINDOW_SECONDS
            ):

                student["changes"].popleft()

            movement_count = len(student["changes"])

            head_position = student["stable_state"]

            # ----------------------------------
            # F. ARM MOVEMENT ANALYSIS
            # ----------------------------------

            left_raised = (
                left_wrist[1] < left_shoulder[1] - 20
            )

            right_raised = (
                right_wrist[1] < right_shoulder[1] - 20
            )

            left_extension = (
                abs(left_wrist[0] - left_shoulder[0])
                / shoulder_width
            )

            right_extension = (
                abs(right_wrist[0] - right_shoulder[0])
                / shoulder_width
            )

            arm_observations = []

            if left_raised:
                arm_observations.append("LEFT HAND RAISED")

            if right_raised:
                arm_observations.append("RIGHT HAND RAISED")

            if left_extension > 1.0:
                arm_observations.append("LEFT ARM EXTENDED")

            if right_extension > 1.0:
                arm_observations.append("RIGHT ARM EXTENDED")

            if arm_observations:
                arm_label = ", ".join(arm_observations)
            else:
                arm_label = "NO RAISED/EXTENDED ARM OBSERVED"

            # ----------------------------------
            # G. ASSOCIATE MOBILE PHONE
            # ----------------------------------

            px1, py1, px2, py2 = boxes[i]

            phone_detected = False

            for phone in phones:

                fx1, fy1, fx2, fy2 = phone["box"]

                phone_center_x = (fx1 + fx2) / 2
                phone_center_y = (fy1 + fy2) / 2

                if (
                    px1 <= phone_center_x <= px2
                    and py1 <= phone_center_y <= py2
                ):

                    phone_detected = True

            # ----------------------------------
            # H. GENERATE MONITORING STATUS
            # ----------------------------------

            if (
                movement_count >= MOVEMENT_THRESHOLD
                or phone_detected
            ):

                status = "REVIEW"

            else:

                status = "OBSERVATION"

            # Arm movements are recorded separately.
            # They do not automatically generate a review alert.

            # ----------------------------------
            # I. DISPLAY INFORMATION
            # ----------------------------------

            x1 = int(px1)
            y1 = int(py1)

            color = (
                (0, 165, 255)
                if status == "REVIEW"
                else (0, 255, 0)
            )

            label1 = (
                f"ID {track_id} | HEAD: {head_position}"
            )

            label2 = (
                f"Head changes: {movement_count}"
            )

            label3 = (
                f"Arm: {arm_label}"
            )

            label4 = (
                f"Phone: {'YES' if phone_detected else 'NO'}"
            )

            label5 = f"Status: {status}"

            labels = [
                label1,
                label2,
                label3,
                label4,
                label5
            ]

            for j, label in enumerate(labels):

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 10 - (j * 20))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2
                )

            # ----------------------------------
            # J. SAVE STUDENT OBSERVATIONS TO CSV
            # ----------------------------------

            writer.writerow({

                "Timestamp": round(video_time, 2),

                "Student_ID": track_id,

                "Head_Position": head_position,

                "Head_Movement_Count": movement_count,

                "Arm_Observation": arm_label,

                "Phone_Detected": (
                    "YES" if phone_detected else "NO"
                ),

                "Monitoring_Status": status

            })

    # --------------------------------------
    # K. DRAW PHONE DETECTIONS
    # --------------------------------------

    for phone in phones:

        x1, y1, x2, y2 = phone["box"]

        confidence = phone["confidence"]

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"PHONE {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # --------------------------------------
    # L. DISPLAY AND SAVE OUTPUT VIDEO
    # --------------------------------------

    cv2.putText(
        annotated_frame,
        "AI EXAM MONITORING SYSTEM",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Integrated AI Exam Monitoring System",
        annotated_frame
    )

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 6. RELEASE RESOURCES
# ==========================================

cap.release()
out.release()

csv_file.close()

cv2.destroyAllWindows()

print("\n================================")
print("INTEGRATED MONITORING COMPLETED")
print("================================")

print("Total frames processed:", frame_number)

print("Total phone detections:", total_phone_detections)

print("Output video:", output_path)

print("CSV report:", csv_path)
