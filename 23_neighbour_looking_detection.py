
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

OUTPUT_VIDEO = "outputs/neighbour_looking_detection.mp4"
OUTPUT_CSV = "outputs/neighbour_looking_report.csv"

MODEL_PATH = "yolo11n-pose.pt"

CONFIDENCE = 0.35

# Head orientation thresholds
LEFT_THRESHOLD = -0.20
RIGHT_THRESHOLD = 0.20

# Smoothing factor
SMOOTHING_ALPHA = 0.25

# Minimum continuous duration to generate an event
MIN_LOOKING_DURATION = 1.5

# Same-row tolerance
SAME_ROW_TOLERANCE = 0.55

# Maximum distance to consider another student
MAX_NEIGHBOUR_DISTANCE = 2.5

# Change this to False if left/right direction appears inverted
LEFT_POINTS_TO_IMAGE_LEFT = True

# Minimum keypoint confidence
KEYPOINT_CONFIDENCE = 0.25


# ---------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------
# 3. LOAD YOLO POSE MODEL
# ---------------------------------------------------------

print("Loading YOLO11 Pose model...")

model = YOLO(MODEL_PATH)

print("YOLO11 Pose model loaded successfully.")


# ---------------------------------------------------------
# 4. OPEN INPUT VIDEO
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Unable to open input video.")
    print("Check whether the video exists at:", VIDEO_PATH)
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
# 5. CREATE OUTPUT VIDEO WRITER
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# 6. STUDENT TRACKING HISTORY
# ---------------------------------------------------------

student_history = defaultdict(lambda: {
    "smoothed_deviation": None,
    "direction": "UNCERTAIN",
    "candidate_neighbour": None,
    "candidate_direction": None,
    "candidate_start": None,
    "active_event": None
})


# Store completed events
events = []


# ---------------------------------------------------------
# 7. FUNCTION TO ESTIMATE HEAD DIRECTION
# ---------------------------------------------------------

def estimate_head_direction(keypoints, keypoint_conf):

    # COCO keypoint indices:
    # 0 = Nose
    # 1 = Left Eye
    # 2 = Right Eye

    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    # Verify keypoint confidence
    if keypoint_conf is not None:

        if (
            keypoint_conf[0] < KEYPOINT_CONFIDENCE
            or keypoint_conf[1] < KEYPOINT_CONFIDENCE
            or keypoint_conf[2] < KEYPOINT_CONFIDENCE
        ):
            return None

    # Eye centre
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0

    # Distance between eyes
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance < 1:
        return None

    # Normalized horizontal deviation
    deviation = (nose[0] - eye_center_x) / eye_distance

    # Determine direction
    if LEFT_POINTS_TO_IMAGE_LEFT:

        if deviation < LEFT_THRESHOLD:
            direction = "IMAGE-LEFT"

        elif deviation > RIGHT_THRESHOLD:
            direction = "IMAGE-RIGHT"

        else:
            direction = "FRONT"

    else:

        if deviation > RIGHT_THRESHOLD:
            direction = "IMAGE-LEFT"

        elif deviation < LEFT_THRESHOLD:
            direction = "IMAGE-RIGHT"

        else:
            direction = "FRONT"

    return deviation, direction


# ---------------------------------------------------------
# 8. FUNCTION TO FIND A NEIGHBOURING STUDENT
# ---------------------------------------------------------

def find_neighbour(student_id, student, people):

    direction = student["direction"]

    if direction not in ["IMAGE-LEFT", "IMAGE-RIGHT"]:
        return None

    student_x, student_y = student["center"]

    student_width = student["width"]
    student_height = student["height"]

    possible_neighbours = []

    for other_id, other in people.items():

        if other_id == student_id:
            continue

        other_x, other_y = other["center"]

        # Check whether both students are approximately
        # in the same horizontal row
        vertical_difference = abs(student_y - other_y)

        row_limit = SAME_ROW_TOLERANCE * max(
            student_height,
            other["height"]
        )

        if vertical_difference > row_limit:
            continue

        horizontal_difference = other_x - student_x

        # Check direction
        if direction == "IMAGE-LEFT":

            if horizontal_difference >= 0:
                continue

        elif direction == "IMAGE-RIGHT":

            if horizontal_difference <= 0:
                continue

        distance = abs(horizontal_difference)

        # Maximum neighbour distance
        maximum_distance = MAX_NEIGHBOUR_DISTANCE * student_width

        if distance > maximum_distance:
            continue

        possible_neighbours.append(
            (distance, other_id)
        )

    if not possible_neighbours:
        return None

    # Select nearest student in the estimated direction
    possible_neighbours.sort(key=lambda item: item[0])

    return possible_neighbours[0][1]


# ---------------------------------------------------------
# 9. FUNCTION TO CLOSE AN ACTIVE EVENT
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
# 10. MAIN VIDEO PROCESSING LOOP
# ---------------------------------------------------------

print("\nStarting neighbour-looking analysis...\n")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

    # -----------------------------------------------------
    # YOLO11 POSE DETECTION AND BYTETRACK TRACKING
    # -----------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    people = {}

    # -----------------------------------------------------
    # EXTRACT STUDENT INFORMATION
    # -----------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
        and result.keypoints is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = result.boxes.id.cpu().numpy().astype(int)

        keypoints_array = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoint_conf_array = (
                result.keypoints.conf.cpu().numpy()
            )
        else:
            keypoint_conf_array = None

        for index, track_id in enumerate(track_ids):

            x1, y1, x2, y2 = boxes[index]

            # Bounding box centre
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            student_width = max(1.0, x2 - x1)
            student_height = max(1.0, y2 - y1)

            keypoints = keypoints_array[index]

            if keypoint_conf_array is not None:
                keypoint_conf = keypoint_conf_array[index]
            else:
                keypoint_conf = None

            # Estimate head direction
            head_result = estimate_head_direction(
                keypoints,
                keypoint_conf
            )

            if head_result is None:
                continue

            deviation, direction = head_result

            # -------------------------------------------------
            # SMOOTH HEAD ORIENTATION
            # -------------------------------------------------

            state = student_history[track_id]

            previous_deviation = state["smoothed_deviation"]

            if previous_deviation is None:

                smoothed_deviation = deviation

            else:

                smoothed_deviation = (
                    SMOOTHING_ALPHA * deviation
                    + (1 - SMOOTHING_ALPHA) * previous_deviation
                )

            state["smoothed_deviation"] = smoothed_deviation

            # Recalculate direction after smoothing
            if LEFT_POINTS_TO_IMAGE_LEFT:

                if smoothed_deviation < LEFT_THRESHOLD:
                    direction = "IMAGE-LEFT"

                elif smoothed_deviation > RIGHT_THRESHOLD:
                    direction = "IMAGE-RIGHT"

                else:
                    direction = "FRONT"

            else:

                if smoothed_deviation > RIGHT_THRESHOLD:
                    direction = "IMAGE-LEFT"

                elif smoothed_deviation < LEFT_THRESHOLD:
                    direction = "IMAGE-RIGHT"

                else:
                    direction = "FRONT"

            state["direction"] = direction

            # Store student information
            people[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "width": student_width,
                "height": student_height,
                "direction": direction
            }

    # -----------------------------------------------------
    # FIND NEIGHBOURING STUDENTS
    # -----------------------------------------------------

    for track_id, student in people.items():

        state = student_history[track_id]

        direction = student["direction"]

        neighbour_id = find_neighbour(
            track_id,
            student,
            people
        )

        # No neighbour in the estimated direction
        if neighbour_id is None:

            close_event(state, current_time)

            state["candidate_neighbour"] = None
            state["candidate_direction"] = None
            state["candidate_start"] = None

            continue

        # -------------------------------------------------
        # CHECK WHETHER THE SAME NEIGHBOUR IS BEING OBSERVED
        # -------------------------------------------------

        if (
            state["candidate_neighbour"] != neighbour_id
            or state["candidate_direction"] != direction
        ):

            # Close previous event if the target changes
            close_event(state, current_time)

            state["candidate_neighbour"] = neighbour_id
            state["candidate_direction"] = direction
            state["candidate_start"] = current_time

        # Calculate duration
        looking_duration = (
            current_time - state["candidate_start"]
        )

        # -------------------------------------------------
        # GENERATE EVENT AFTER MINIMUM DURATION
        # -------------------------------------------------

        if looking_duration >= MIN_LOOKING_DURATION:

            if state["active_event"] is None:

                state["active_event"] = {
                    "Student_ID": track_id,
                    "Neighbour_Track_ID": neighbour_id,
                    "Direction": direction,
                    "start_time": round(
                        state["candidate_start"], 2
                    ),
                    "end_time": round(current_time, 2),
                    "duration_seconds": round(
                        looking_duration, 2
                    ),
                    "Observation": "SUSTAINED NEIGHBOUR-FACING"
                }

            else:

                # Update the active event duration
                state["active_event"]["end_time"] = round(
                    current_time, 2
                )

                state["active_event"]["duration_seconds"] = round(
                    looking_duration, 2
                )

    # -----------------------------------------------------
    # DRAW STUDENT BOUNDING BOXES AND INFORMATION
    # -----------------------------------------------------

    for track_id, student in people.items():

        x1, y1, x2, y2 = student["bbox"]

        state = student_history[track_id]

        direction = student["direction"]

        neighbour_id = state["candidate_neighbour"]

        active_event = state["active_event"]

        # Convert all coordinates to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Default colour
        color = (0, 255, 0)

        label = f"ID: {track_id} | {direction}"

        # Show sustained neighbour-facing observation
        if active_event is not None:

            color = (0, 0, 255)

            duration = active_event["duration_seconds"]

            label = (
                f"ID: {track_id} | REVIEW | "
                f"{duration:.1f}s"
            )

        elif neighbour_id is not None:

            color = (0, 255, 255)

            label = (
                f"ID: {track_id} | "
                f"Facing ID: {neighbour_id}"
            )

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Display student information
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

    # Save processed frame
    out.write(frame)

    # Display processed frame
    cv2.imshow(
        "Neighbour Looking Detection",
        frame
    )

    # Press Q to stop processing
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
# 13. SAVE EVENTS TO CSV REPORT
# ---------------------------------------------------------

fieldnames = [
    "Student_ID",
    "Neighbour_Track_ID",
    "Direction",
    "start_time",
    "end_time",
    "duration_seconds",
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
print("NEIGHBOUR LOOKING ANALYSIS COMPLETED")
print("--------------------------------------")

print("Total frames processed:", frame_number)

print("Total sustained neighbour-facing events:", len(events))

print("\nOutput Files:")

print("Video:", OUTPUT_VIDEO)

print("CSV Report:", OUTPUT_CSV)

print("\nAnalysis completed successfully.")
