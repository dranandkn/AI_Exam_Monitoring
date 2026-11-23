
import cv2
import csv
import os
from collections import defaultdict, deque
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

VIDEO_PATH = "videos/recorded_video.mp4"

MODEL_PATH = "yolo11n-pose.pt"

OUTPUT_VIDEO = "outputs/frequent_looking_around.mp4"

OUTPUT_CSV = "outputs/frequent_looking_report.csv"

CONFIDENCE = 0.35

# Head direction thresholds
LEFT_THRESHOLD = -0.20
RIGHT_THRESHOLD = 0.20

# Head direction smoothing
SMOOTHING_ALPHA = 0.25

# Minimum duration for a direction to be considered stable
MIN_DIRECTION_DURATION = 0.5

# Time window for counting direction changes
OBSERVATION_WINDOW = 15.0

# Minimum direction changes required for an observation
MIN_DIRECTION_CHANGES = 4

# Minimum keypoint confidence
KEYPOINT_CONFIDENCE = 0.25

# Change to False if left/right direction is inverted
LEFT_POINTS_TO_IMAGE_LEFT = True


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
# 6. STUDENT MOVEMENT HISTORY
# ---------------------------------------------------------

student_history = defaultdict(lambda: {

    "smoothed_deviation": None,

    "current_direction": None,

    "direction_start": None,

    "last_confirmed_direction": None,

    "direction_changes": deque(),

    "active_event": None,

    "total_changes": 0

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


    # Check keypoint confidence
    if keypoint_conf is not None:

        required_points = [0, 1, 2]

        for point_index in required_points:

            if keypoint_conf[point_index] < KEYPOINT_CONFIDENCE:

                return None


    # Calculate eye centre
    eye_center_x = (
        left_eye[0] + right_eye[0]
    ) / 2.0


    # Calculate distance between eyes
    eye_distance = abs(
        right_eye[0] - left_eye[0]
    )


    if eye_distance < 1:

        return None


    # Normalized nose deviation
    deviation = (
        nose[0] - eye_center_x
    ) / eye_distance


    return deviation


# ---------------------------------------------------------
# 8. FUNCTION TO CLASSIFY HEAD DIRECTION
# ---------------------------------------------------------

def classify_direction(deviation):

    if LEFT_POINTS_TO_IMAGE_LEFT:

        if deviation < LEFT_THRESHOLD:

            return "IMAGE-LEFT"

        elif deviation > RIGHT_THRESHOLD:

            return "IMAGE-RIGHT"

        else:

            return "FRONT"


    else:

        if deviation > RIGHT_THRESHOLD:

            return "IMAGE-LEFT"

        elif deviation < LEFT_THRESHOLD:

            return "IMAGE-RIGHT"

        else:

            return "FRONT"


# ---------------------------------------------------------
# 9. FUNCTION TO CLOSE AN ACTIVE EVENT
# ---------------------------------------------------------

def close_event(state, current_time):

    active_event = state["active_event"]

    if active_event is None:

        return


    active_event["end_time"] = round(
        current_time, 2
    )

    active_event["duration_seconds"] = round(
        current_time - active_event["start_time"],
        2
    )

    events.append(active_event.copy())

    state["active_event"] = None


# ---------------------------------------------------------
# 10. MAIN VIDEO PROCESSING LOOP
# ---------------------------------------------------------

print("\nStarting frequent looking-around analysis...\n")

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break


    frame_number += 1

    current_time = frame_number / fps


    # -----------------------------------------------------
    # 10A. YOLO11 POSE DETECTION AND TRACKING
    # -----------------------------------------------------

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONFIDENCE,

        verbose=False

    )

    result = results[0]


    # Store students detected in the current frame
    current_students = set()


    # -----------------------------------------------------
    # 10B. EXTRACT STUDENT INFORMATION
    # -----------------------------------------------------

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
        # PROCESS EACH TRACKED STUDENT
        # -------------------------------------------------

        for index, track_id in enumerate(track_ids):

            current_students.add(track_id)


            x1, y1, x2, y2 = boxes[index]

            keypoints = keypoints_array[index]


            if keypoint_conf_array is not None:

                keypoint_conf = keypoint_conf_array[index]

            else:

                keypoint_conf = None


            # -------------------------------------------------
            # ESTIMATE HEAD DIRECTION
            # -------------------------------------------------

            deviation = estimate_head_direction(

                keypoints,

                keypoint_conf

            )


            if deviation is None:

                continue


            # Get student history
            state = student_history[track_id]


            # -------------------------------------------------
            # SMOOTH HEAD ORIENTATION
            # -------------------------------------------------

            previous_deviation = (
                state["smoothed_deviation"]
            )


            if previous_deviation is None:

                smoothed_deviation = deviation

            else:

                smoothed_deviation = (

                    SMOOTHING_ALPHA * deviation

                    + (1 - SMOOTHING_ALPHA)
                    * previous_deviation

                )


            state["smoothed_deviation"] = (
                smoothed_deviation
            )


            # Classify direction
            direction = classify_direction(
                smoothed_deviation
            )


            # -------------------------------------------------
            # DETECT DIRECTION CHANGES
            # -------------------------------------------------

            previous_direction = (
                state["current_direction"]
            )


            if previous_direction is None:

                state["current_direction"] = direction

                state["direction_start"] = current_time


            elif direction != previous_direction:

                previous_duration = (

                    current_time
                    - state["direction_start"]

                )


                # Count only sustained directions
                if previous_duration >= MIN_DIRECTION_DURATION:

                    state["last_confirmed_direction"] = (
                        previous_direction
                    )

                    state["direction_changes"].append(
                        current_time
                    )

                    state["total_changes"] += 1


                # Start tracking the new direction
                state["current_direction"] = direction

                state["direction_start"] = current_time


            # -------------------------------------------------
            # REMOVE OLD DIRECTION CHANGES
            # -------------------------------------------------

            while (

                state["direction_changes"]

                and current_time
                - state["direction_changes"][0]
                > OBSERVATION_WINDOW

            ):

                state["direction_changes"].popleft()


            # Current number of direction changes
            change_count = len(
                state["direction_changes"]
            )


            # -------------------------------------------------
            # CHECK FREQUENT LOOKING CONDITION
            # -------------------------------------------------

            if change_count >= MIN_DIRECTION_CHANGES:

                # Create a new observation
                if state["active_event"] is None:

                    state["active_event"] = {

                        "Student_ID": track_id,

                        "start_time": round(
                            state["direction_changes"][0], 2
                        ),

                        "end_time": round(
                            current_time, 2
                        ),

                        "duration_seconds": round(

                            current_time
                            - state["direction_changes"][0],

                            2

                        ),

                        "direction_changes": change_count,

                        "Observation": "FREQUENT HEAD DIRECTION CHANGES"

                    }


                else:

                    # Update existing observation
                    state["active_event"]["end_time"] = round(
                        current_time, 2
                    )

                    state["active_event"]["duration_seconds"] = round(

                        current_time
                        - state["active_event"]["start_time"],

                        2

                    )

                    state["active_event"]["direction_changes"] = (
                        change_count
                    )


            else:

                # Close event when the count drops below threshold
                close_event(
                    state,
                    current_time
                )


            # -------------------------------------------------
            # DRAW STUDENT INFORMATION
            # -------------------------------------------------

            x1 = int(x1)

            y1 = int(y1)

            x2 = int(x2)

            y2 = int(y2)


            color = (0, 255, 0)


            label = (
                f"ID: {track_id} | {direction}"
            )


            if state["active_event"] is not None:

                color = (0, 0, 255)

                label = (

                    f"ID: {track_id} | "
                    f"FREQUENT LOOKING | "
                    f"{change_count} changes"

                )


            elif change_count > 0:

                color = (0, 255, 255)

                label = (

                    f"ID: {track_id} | "
                    f"Changes: {change_count}"

                )


            # Draw bounding box
            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                color,

                2

            )


            # Display information
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
    # 10C. HANDLE STUDENTS WHO DISAPPEAR FROM FRAME
    # -----------------------------------------------------

    for track_id, state in student_history.items():

        if (

            track_id not in current_students

            and state["active_event"] is not None

        ):

            close_event(
                state,
                current_time
            )


    # -----------------------------------------------------
    # 10D. DISPLAY FRAME INFORMATION
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
    # 10E. SAVE AND DISPLAY VIDEO FRAME
    # -----------------------------------------------------

    out.write(frame)


    cv2.imshow(

        "Frequent Looking Around Detection",

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
# 11. CLOSE REMAINING ACTIVE EVENTS
# ---------------------------------------------------------

final_time = frame_number / fps


for track_id, state in student_history.items():

    close_event(
        state,
        final_time
    )


# ---------------------------------------------------------
# 12. RELEASE VIDEO RESOURCES
# ---------------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# 13. SAVE EVENTS TO CSV
# ---------------------------------------------------------

fieldnames = [

    "Student_ID",

    "start_time",

    "end_time",

    "duration_seconds",

    "direction_changes",

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

print("FREQUENT LOOKING ANALYSIS COMPLETED")

print("--------------------------------------")

print("Total frames processed:", frame_number)

print("Total observations:", len(events))

print("\nOutput Files:")

print("Video:", OUTPUT_VIDEO)

print("CSV Report:", OUTPUT_CSV)

print("\nAnalysis completed successfully.")
