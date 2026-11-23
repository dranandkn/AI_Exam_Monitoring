
import cv2
import csv
import os
import math
from itertools import combinations
from collections import defaultdict
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = "videos/recorded_video.mp4"

MODEL_PATH = "yolo11n-pose.pt"

OUTPUT_VIDEO = "outputs/note_passing_interaction_detection.mp4"

OUTPUT_CSV = "outputs/student_interaction_report.csv"

CONFIDENCE = 0.35

KEYPOINT_CONFIDENCE = 0.25

# Minimum hand-to-hand proximity duration
MIN_INTERACTION_DURATION = 0.35

# Maximum distance between two wrists,
# expressed as a multiple of average shoulder width
WRIST_DISTANCE_FACTOR = 0.65

# Minimum arm extension relative to shoulder width
ARM_EXTENSION_FACTOR = 0.90

# Students must be approximately in the same row
SAME_ROW_TOLERANCE = 0.65

# Maximum horizontal separation between neighbouring students
MAX_NEIGHBOUR_DISTANCE = 3.0


# ---------------------------------------------------------
# 2. CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------
# 3. LOAD YOLO11 POSE MODEL
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

    print("Check the video path:", VIDEO_PATH)

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
# 6. INTERACTION HISTORY
# ---------------------------------------------------------

interaction_history = defaultdict(lambda: {

    "candidate_start": None,

    "active_event": None

})


# Completed interaction events
events = []


# ---------------------------------------------------------
# 7. DISTANCE CALCULATION
# ---------------------------------------------------------

def calculate_distance(point1, point2):

    dx = point1[0] - point2[0]

    dy = point1[1] - point2[1]

    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------
# 8. CLOSE AN ACTIVE INTERACTION EVENT
# ---------------------------------------------------------

def close_event(pair_key, state, current_time):

    active_event = state["active_event"]

    if active_event is None:
        return

    active_event["end_time"] = round(
        current_time, 2
    )

    active_event["duration_seconds"] = round(
        current_time - active_event["start_time"], 2
    )

    events.append(active_event.copy())

    state["active_event"] = None

    state["candidate_start"] = None


# ---------------------------------------------------------
# 9. EXTRACT STUDENT BODY KEYPOINTS
# ---------------------------------------------------------

def extract_student_information(
    bbox,
    keypoints,
    keypoint_conf
):

    # COCO keypoint indices:
    #
    # 5  = Left shoulder
    # 6  = Right shoulder
    # 7  = Left elbow
    # 8  = Right elbow
    # 9  = Left wrist
    # 10 = Right wrist

    required_points = [5, 6, 7, 8, 9, 10]

    if keypoint_conf is None:
        return None

    for point_index in required_points:

        if keypoint_conf[point_index] < KEYPOINT_CONFIDENCE:

            return None

    left_shoulder = tuple(keypoints[5])

    right_shoulder = tuple(keypoints[6])

    left_elbow = tuple(keypoints[7])

    right_elbow = tuple(keypoints[8])

    left_wrist = tuple(keypoints[9])

    right_wrist = tuple(keypoints[10])


    # Calculate shoulder width
    shoulder_width = calculate_distance(
        left_shoulder,
        right_shoulder
    )

    if shoulder_width < 5:
        return None


    # Check whether either arm is extended
    left_arm_length = calculate_distance(
        left_shoulder,
        left_wrist
    )

    right_arm_length = calculate_distance(
        right_shoulder,
        right_wrist
    )

    left_arm_extended = (
        left_arm_length
        >= ARM_EXTENSION_FACTOR * shoulder_width
    )

    right_arm_extended = (
        right_arm_length
        >= ARM_EXTENSION_FACTOR * shoulder_width
    )


    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2.0

    center_y = (y1 + y2) / 2.0


    return {

        "bbox": bbox,

        "center": (center_x, center_y),

        "shoulder_width": shoulder_width,

        "left_shoulder": left_shoulder,

        "right_shoulder": right_shoulder,

        "left_elbow": left_elbow,

        "right_elbow": right_elbow,

        "left_wrist": left_wrist,

        "right_wrist": right_wrist,

        "left_arm_extended": left_arm_extended,

        "right_arm_extended": right_arm_extended,

        "wrists": [

            left_wrist,

            right_wrist

        ]

    }


# ---------------------------------------------------------
# 10. CHECK INTERACTION BETWEEN TWO STUDENTS
# ---------------------------------------------------------

def check_student_interaction(student_a, student_b):

    # Check whether students are approximately in the same row
    vertical_difference = abs(
        student_a["center"][1]
        - student_b["center"][1]
    )

    row_limit = SAME_ROW_TOLERANCE * max(

        student_a["bbox"][3] - student_a["bbox"][1],

        student_b["bbox"][3] - student_b["bbox"][1]

    )

    if vertical_difference > row_limit:

        return None


    # Check horizontal separation
    horizontal_difference = abs(

        student_a["center"][0]
        - student_b["center"][0]

    )

    average_width = (

        student_a["shoulder_width"]
        + student_b["shoulder_width"]

    ) / 2.0

    if horizontal_difference > (
        MAX_NEIGHBOUR_DISTANCE * average_width
    ):

        return None


    # Find closest wrist pair
    closest_distance = float("inf")

    closest_wrist_a = None

    closest_wrist_b = None

    for wrist_a in student_a["wrists"]:

        for wrist_b in student_b["wrists"]:

            distance = calculate_distance(
                wrist_a,
                wrist_b
            )

            if distance < closest_distance:

                closest_distance = distance

                closest_wrist_a = wrist_a

                closest_wrist_b = wrist_b


    # Distance threshold
    maximum_distance = (
        WRIST_DISTANCE_FACTOR * average_width
    )

    if closest_distance > maximum_distance:

        return None


    # Check whether at least one student has an extended arm
    arm_extension_detected = (

        student_a["left_arm_extended"]

        or student_a["right_arm_extended"]

        or student_b["left_arm_extended"]

        or student_b["right_arm_extended"]

    )

    if not arm_extension_detected:

        return None


    return {

        "distance": closest_distance,

        "wrist_a": closest_wrist_a,

        "wrist_b": closest_wrist_b

    }


# ---------------------------------------------------------
# 11. MAIN VIDEO PROCESSING LOOP
# ---------------------------------------------------------

print("\nStarting student interaction analysis...\n")

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_number += 1

    current_time = frame_number / fps


    # -----------------------------------------------------
    # 11A. YOLO POSE DETECTION AND TRACKING
    # -----------------------------------------------------

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONFIDENCE,

        verbose=False

    )

    result = results[0]


    # Store information about detected students
    people = {}


    if (

        result.boxes is not None

        and result.boxes.id is not None

        and result.keypoints is not None

    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (

            result.boxes.id.cpu().numpy().astype(int)

        )

        keypoints_array = (

            result.keypoints.xy.cpu().numpy()

        )


        if result.keypoints.conf is not None:

            keypoint_conf_array = (

                result.keypoints.conf.cpu().numpy()

            )

        else:

            keypoint_conf_array = None


        # -------------------------------------------------
        # 11B. EXTRACT STUDENT INFORMATION
        # -------------------------------------------------

        for index, track_id in enumerate(track_ids):

            bbox = boxes[index]

            keypoints = keypoints_array[index]


            if keypoint_conf_array is not None:

                keypoint_conf = keypoint_conf_array[index]

            else:

                keypoint_conf = None


            student_information = extract_student_information(

                bbox,

                keypoints,

                keypoint_conf

            )


            if student_information is None:

                continue


            people[track_id] = student_information


    # -----------------------------------------------------
    # 11C. DRAW STUDENT BOUNDING BOXES
    # -----------------------------------------------------

    for track_id, student in people.items():

        x1, y1, x2, y2 = student["bbox"]

        x1 = int(x1)

        y1 = int(y1)

        x2 = int(x2)

        y2 = int(y2)


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2

        )


        cv2.putText(

            frame,

            f"Student ID: {track_id}",

            (int(x1), max(20, int(y1) - 10)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2

        )


        # Draw wrist points
        for wrist in student["wrists"]:

            wrist_x = int(wrist[0])

            wrist_y = int(wrist[1])

            cv2.circle(

                frame,

                (wrist_x, wrist_y),

                5,

                (255, 0, 255),

                -1

            )


    # -----------------------------------------------------
    # 11D. CHECK INTERACTIONS BETWEEN STUDENT PAIRS
    # -----------------------------------------------------

    active_pairs = set()

    student_pairs = list(
        combinations(people.keys(), 2)
    )


    for student_id_a, student_id_b in student_pairs:

        student_a = people[student_id_a]

        student_b = people[student_id_b]


        interaction = check_student_interaction(

            student_a,

            student_b

        )


        if interaction is None:

            continue


        # Create consistent pair identifier
        pair_key = tuple(
            sorted((student_id_a, student_id_b))
        )

        active_pairs.add(pair_key)

        state = interaction_history[pair_key]


        # -------------------------------------------------
        # 11E. START OR CONTINUE INTERACTION
        # -------------------------------------------------

        if state["candidate_start"] is None:

            state["candidate_start"] = current_time


        interaction_duration = (

            current_time - state["candidate_start"]

        )


        # -------------------------------------------------
        # 11F. RECORD SUSTAINED INTERACTION
        # -------------------------------------------------

        if interaction_duration >= MIN_INTERACTION_DURATION:

            if state["active_event"] is None:

                state["active_event"] = {

                    "Student_ID_1": pair_key[0],

                    "Student_ID_2": pair_key[1],

                    "start_time": round(
                        state["candidate_start"], 2
                    ),

                    "end_time": round(
                        current_time, 2
                    ),

                    "duration_seconds": round(
                        interaction_duration, 2
                    ),

                    "wrist_distance_pixels": round(
                        interaction["distance"], 2
                    ),

                    "Observation": "POSSIBLE HAND-TO-HAND INTERACTION"

                }


            else:

                state["active_event"]["end_time"] = round(
                    current_time, 2
                )

                state["active_event"]["duration_seconds"] = round(
                    interaction_duration, 2
                )

                state["active_event"]["wrist_distance_pixels"] = round(
                    interaction["distance"], 2
                )


            # Draw line between the two detected wrists
            wrist_a = interaction["wrist_a"]

            wrist_b = interaction["wrist_b"]


            cv2.line(

                frame,

                (int(wrist_a[0]), int(wrist_a[1])),

                (int(wrist_b[0]), int(wrist_b[1])),

                (0, 0, 255),

                3

            )


            # Draw interaction label
            middle_x = int(
                (wrist_a[0] + wrist_b[0]) / 2
            )

            middle_y = int(
                (wrist_a[1] + wrist_b[1]) / 2
            )


            cv2.putText(

                frame,

                "POSSIBLE INTERACTION",

                (middle_x, max(20, middle_y - 15)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 0, 255),

                2

            )


    # -----------------------------------------------------
    # 11G. CLOSE INTERACTIONS THAT HAVE ENDED
    # -----------------------------------------------------

    for pair_key, state in interaction_history.items():

        if (

            pair_key not in active_pairs

            and state["active_event"] is not None

        ):

            close_event(

                pair_key,

                state,

                current_time

            )


        elif (

            pair_key not in active_pairs

            and state["candidate_start"] is not None

        ):

            state["candidate_start"] = None


    # -----------------------------------------------------
    # 11H. DISPLAY FRAME INFORMATION
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


    cv2.putText(

        frame,

        f"Detected Students: {len(people)}",

        (20, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    # -----------------------------------------------------
    # 11I. SAVE AND DISPLAY FRAME
    # -----------------------------------------------------

    out.write(frame)


    cv2.imshow(

        "Student Interaction Detection",

        frame

    )


    # Press Q to stop processing
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


    # Display processing progress
    if frame_number % 100 == 0:

        print(

            f"Processed {frame_number}/{total_frames} frames"

        )


# ---------------------------------------------------------
# 12. CLOSE REMAINING ACTIVE EVENTS
# ---------------------------------------------------------

final_time = frame_number / fps


for pair_key, state in interaction_history.items():

    close_event(

        pair_key,

        state,

        final_time

    )


# ---------------------------------------------------------
# 13. RELEASE VIDEO RESOURCES
# ---------------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# 14. SAVE INTERACTION REPORT
# ---------------------------------------------------------

fieldnames = [

    "Student_ID_1",

    "Student_ID_2",

    "start_time",

    "end_time",

    "duration_seconds",

    "wrist_distance_pixels",

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
# 15. DISPLAY FINAL SUMMARY
# ---------------------------------------------------------

print("\n--------------------------------------")

print("STUDENT INTERACTION ANALYSIS COMPLETED")

print("--------------------------------------")

print("Total frames processed:", frame_number)

print("Total interaction observations:", len(events))

print("\nOutput Files:")

print("Video:", OUTPUT_VIDEO)

print("CSV Report:", OUTPUT_CSV)

print("\nAnalysis completed successfully.")
