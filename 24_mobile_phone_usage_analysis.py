
import cv2
import csv
import os
import math
from collections import defaultdict
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = "videos/recorded_video.mp4"

POSE_MODEL_PATH = "yolo11n-pose.pt"
OBJECT_MODEL_PATH = "yolo11n.pt"

OUTPUT_VIDEO = "outputs/mobile_phone_usage_analysis.mp4"
OUTPUT_CSV = "outputs/mobile_phone_usage_report.csv"

POSE_CONFIDENCE = 0.35
PHONE_CONFIDENCE = 0.35

# COCO class ID for cell phone
PHONE_CLASS_ID = 67

# Minimum duration for a phone-hand observation
MIN_PHONE_DURATION = 1.0

# Wrist-to-phone distance threshold,
# expressed as a multiple of shoulder width
WRIST_DISTANCE_FACTOR = 1.5

# Minimum keypoint confidence
KEYPOINT_CONFIDENCE = 0.25


# ---------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------
# 3. LOAD YOLO MODELS
# ---------------------------------------------------------

print("Loading YOLO11 Pose model...")

pose_model = YOLO(POSE_MODEL_PATH)

print("Pose model loaded successfully.")

print("\nLoading YOLO11 Object Detection model...")

object_model = YOLO(OBJECT_MODEL_PATH)

print("Object detection model loaded successfully.")


# ---------------------------------------------------------
# 4. OPEN INPUT VIDEO
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Unable to open input video.")
    raise SystemExit


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("\nVideo Information")
print("------------------------------")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frames)


# ---------------------------------------------------------
# 5. CREATE OUTPUT VIDEO
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# 6. STUDENT HISTORY
# ---------------------------------------------------------

student_history = defaultdict(lambda: {

    "candidate_start": None,

    "candidate_phone": None,

    "active_event": None

})


# Completed events
events = []


# ---------------------------------------------------------
# 7. DISTANCE CALCULATION FUNCTION
# ---------------------------------------------------------

def calculate_distance(point1, point2):

    dx = point1[0] - point2[0]

    dy = point1[1] - point2[1]

    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------
# 8. FUNCTION TO CLOSE PHONE EVENT
# ---------------------------------------------------------

def close_event(state, current_time):

    active_event = state["active_event"]

    if active_event is None:
        return

    active_event["end_time"] = round(current_time, 2)

    active_event["duration_seconds"] = round(
        current_time - active_event["start_time"],
        2
    )

    events.append(active_event.copy())

    state["active_event"] = None


# ---------------------------------------------------------
# 9. FUNCTION TO FIND PHONE NEAR STUDENT'S WRIST
# ---------------------------------------------------------

def find_phone_near_wrist(
    student_bbox,
    keypoints,
    keypoint_conf,
    phones
):

    x1, y1, x2, y2 = student_bbox

    # COCO pose keypoint indices:
    # 5 = Left shoulder
    # 6 = Right shoulder
    # 9 = Left wrist
    # 10 = Right wrist

    left_shoulder = keypoints[5]

    right_shoulder = keypoints[6]

    left_wrist = keypoints[9]

    right_wrist = keypoints[10]

    # Check keypoint confidence
    if keypoint_conf is None:
        return None

    required_points = [5, 6, 9, 10]

    for point_index in required_points:

        if keypoint_conf[point_index] < KEYPOINT_CONFIDENCE:
            return None

    # Calculate shoulder width
    shoulder_width = calculate_distance(
        left_shoulder,
        right_shoulder
    )

    if shoulder_width < 5:
        return None

    maximum_distance = (
        WRIST_DISTANCE_FACTOR * shoulder_width
    )

    # Check each detected phone
    for phone in phones:

        phone_x, phone_y = phone["center"]

        # Phone must be inside the student's bounding box
        if not (
            x1 <= phone_x <= x2
            and y1 <= phone_y <= y2
        ):
            continue

        # Distance from phone to left wrist
        left_distance = calculate_distance(
            (phone_x, phone_y),
            left_wrist
        )

        # Distance from phone to right wrist
        right_distance = calculate_distance(
            (phone_x, phone_y),
            right_wrist
        )

        nearest_distance = min(
            left_distance,
            right_distance
        )

        # Check whether phone is close to either wrist
        if nearest_distance <= maximum_distance:

            return phone

    return None


# ---------------------------------------------------------
# 10. MAIN VIDEO PROCESSING LOOP
# ---------------------------------------------------------

print("\nStarting mobile phone usage analysis...\n")

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps


    # -----------------------------------------------------
    # 10A. STUDENT DETECTION AND POSE ESTIMATION
    # -----------------------------------------------------

    pose_results = pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=POSE_CONFIDENCE,
        verbose=False
    )

    pose_result = pose_results[0]


    # -----------------------------------------------------
    # 10B. MOBILE PHONE DETECTION
    # -----------------------------------------------------

    object_results = object_model.predict(
        frame,
        conf=PHONE_CONFIDENCE,
        classes=[PHONE_CLASS_ID],
        verbose=False
    )

    object_result = object_results[0]


    # -----------------------------------------------------
    # EXTRACT DETECTED PHONES
    # -----------------------------------------------------

    phones = []

    if object_result.boxes is not None:

        phone_boxes = (
            object_result.boxes.xyxy.cpu().numpy()
        )

        phone_confidences = (
            object_result.boxes.conf.cpu().numpy()
        )

        for index, phone_box in enumerate(phone_boxes):

            px1, py1, px2, py2 = phone_box

            phone_center_x = (px1 + px2) / 2.0

            phone_center_y = (py1 + py2) / 2.0

            phones.append({

                "bbox": (
                    px1,
                    py1,
                    px2,
                    py2
                ),

                "center": (
                    phone_center_x,
                    phone_center_y
                ),

                "confidence": float(
                    phone_confidences[index]
                )

            })


    # -----------------------------------------------------
    # EXTRACT STUDENT INFORMATION
    # -----------------------------------------------------

    people = {}

    if (
        pose_result.boxes is not None
        and pose_result.boxes.id is not None
        and pose_result.keypoints is not None
    ):

        student_boxes = (
            pose_result.boxes.xyxy.cpu().numpy()
        )

        track_ids = (
            pose_result.boxes.id.cpu().numpy().astype(int)
        )

        keypoints_array = (
            pose_result.keypoints.xy.cpu().numpy()
        )

        if pose_result.keypoints.conf is not None:

            keypoint_conf_array = (
                pose_result.keypoints.conf.cpu().numpy()
            )

        else:

            keypoint_conf_array = None


        for index, track_id in enumerate(track_ids):

            x1, y1, x2, y2 = student_boxes[index]

            keypoints = keypoints_array[index]

            if keypoint_conf_array is not None:

                keypoint_conf = keypoint_conf_array[index]

            else:

                keypoint_conf = None

            people[track_id] = {

                "bbox": (x1, y1, x2, y2),

                "keypoints": keypoints,

                "keypoint_conf": keypoint_conf

            }


    # -----------------------------------------------------
    # FIND PHONES ASSOCIATED WITH STUDENTS
    # -----------------------------------------------------

    for track_id, student in people.items():

        state = student_history[track_id]

        student_bbox = student["bbox"]

        keypoints = student["keypoints"]

        keypoint_conf = student["keypoint_conf"]


        detected_phone = find_phone_near_wrist(
            student_bbox,
            keypoints,
            keypoint_conf,
            phones
        )


        # -------------------------------------------------
        # PHONE NOT FOUND NEAR STUDENT'S WRIST
        # -------------------------------------------------

        if detected_phone is None:

            close_event(state, current_time)

            state["candidate_start"] = None

            state["candidate_phone"] = None

            continue


        phone_center = detected_phone["center"]

        phone_confidence = detected_phone["confidence"]


        # -------------------------------------------------
        # START OR CONTINUE PHONE OBSERVATION
        # -------------------------------------------------

        if state["candidate_start"] is None:

            state["candidate_start"] = current_time

            state["candidate_phone"] = phone_center


        phone_duration = (
            current_time - state["candidate_start"]
        )


        # -------------------------------------------------
        # GENERATE EVENT AFTER MINIMUM DURATION
        # -------------------------------------------------

        if phone_duration >= MIN_PHONE_DURATION:

            if state["active_event"] is None:

                state["active_event"] = {

                    "Student_ID": track_id,

                    "start_time": round(
                        state["candidate_start"], 2
                    ),

                    "end_time": round(
                        current_time, 2
                    ),

                    "duration_seconds": round(
                        phone_duration, 2
                    ),

                    "phone_confidence": round(
                        phone_confidence, 3
                    ),

                    "Observation": "PHONE NEAR STUDENT WRIST"

                }

            else:

                state["active_event"]["end_time"] = round(
                    current_time, 2
                )

                state["active_event"]["duration_seconds"] = round(
                    phone_duration, 2
                )

                state["active_event"]["phone_confidence"] = round(
                    phone_confidence, 3
                )


    # -----------------------------------------------------
    # DRAW STUDENT BOUNDING BOXES
    # -----------------------------------------------------

    for track_id, student in people.items():

        x1, y1, x2, y2 = student["bbox"]

        state = student_history[track_id]

        x1 = int(x1)

        y1 = int(y1)

        x2 = int(x2)

        y2 = int(y2)


        color = (0, 255, 0)

        label = f"Student ID: {track_id}"


        if state["active_event"] is not None:

            color = (0, 0, 255)

            duration = (
                state["active_event"]["duration_seconds"]
            )

            label = (
                f"ID: {track_id} | "
                f"PHONE REVIEW | {duration:.1f}s"
            )

        elif state["candidate_start"] is not None:

            color = (0, 255, 255)

            label = f"ID: {track_id} | Checking phone"


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (int(x1), max(20, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


    # -----------------------------------------------------
    # DRAW PHONE BOUNDING BOXES
    # -----------------------------------------------------

    for phone in phones:

        px1, py1, px2, py2 = phone["bbox"]

        px1 = int(px1)

        py1 = int(py1)

        px2 = int(px2)

        py2 = int(py2)

        phone_label = "Mobile Phone"

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            phone_label,
            (int(px1), max(20, int(py1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )


    # -----------------------------------------------------
    # DISPLAY FRAME INFORMATION
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Time: {current_time:.2f} sec",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # SAVE AND DISPLAY FRAME
    # -----------------------------------------------------

    out.write(frame)

    cv2.imshow(
        "Mobile Phone Usage Analysis",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


    # Progress information
    if frame_number % 100 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ---------------------------------------------------------
# 11. CLOSE REMAINING ACTIVE EVENTS
# ---------------------------------------------------------

final_time = frame_number / fps

for track_id, state in student_history.items():

    close_event(state, final_time)


# ---------------------------------------------------------
# 12. RELEASE VIDEO RESOURCES
# ---------------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# 13. SAVE CSV REPORT
# ---------------------------------------------------------

fieldnames = [

    "Student_ID",

    "start_time",

    "end_time",

    "duration_seconds",

    "phone_confidence",

    "Observation"

]


with open(
    OUTPUT_CSV,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for event in events:

        writer.writerow(event)


# ---------------------------------------------------------
# 14. DISPLAY FINAL SUMMARY
# ---------------------------------------------------------

print("\n--------------------------------------")

print("MOBILE PHONE ANALYSIS COMPLETED")

print("--------------------------------------")

print("Total frames processed:", frame_number)

print("Total phone-related observations:", len(events))

print("\nOutput Files:")

print("Video:", OUTPUT_VIDEO)

print("CSV Report:", OUTPUT_CSV)

print("\nAnalysis completed successfully.")
