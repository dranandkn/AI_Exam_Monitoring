
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Exam Monitoring System",
    page_icon="🎓",
    layout="wide"
)

st.title("AI-Based Examination Monitoring System")

st.write(
    "Upload an examination video. The system tracks students "
    "and highlights sustained head-turning observations."
)

st.warning(
    "A head turn is only a visual observation. It does not prove "
    "that a student is copying or cheating."
)


# ==========================================
# 2. LOAD YOLO POSE MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = YOLO("yolo11n-pose.pt")

    return model


# ==========================================
# 3. DETECTION SETTINGS
# ==========================================

MIN_TURN_DURATION = 1.0

# Head deviation threshold.
# Smaller values make detection more sensitive.
HEAD_TURN_THRESHOLD = 0.18

# Minimum keypoint confidence
KEYPOINT_CONFIDENCE = 0.30

# Tracking confidence
DETECTION_CONFIDENCE = 0.35

# Change to False if left/right appears reversed
LEFT_POINTS_TO_IMAGE_LEFT = True


# ==========================================
# 4. CALCULATE HEAD DIRECTION
# ==========================================

def calculate_head_direction(points, confidence):

    nose = points[0]
    left_eye = points[1]
    right_eye = points[2]
    left_ear = points[3]
    right_ear = points[4]

    nose_conf = confidence[0]
    left_eye_conf = confidence[1]
    right_eye_conf = confidence[2]
    left_ear_conf = confidence[3]
    right_ear_conf = confidence[4]

    direction = "UNCERTAIN"
    deviation = None

    # Method 1: Use both ears and nose
    if (
        nose_conf > KEYPOINT_CONFIDENCE
        and left_ear_conf > KEYPOINT_CONFIDENCE
        and right_ear_conf > KEYPOINT_CONFIDENCE
    ):

        ear_center = (
            left_ear[0] + right_ear[0]
        ) / 2

        ear_distance = abs(
            left_ear[0] - right_ear[0]
        )

        if ear_distance > 5:

            deviation = (
                nose[0] - ear_center
            ) / ear_distance

    # Method 2: If ears are not both visible,
    # use the nose and visible eye positions.
    elif (
        nose_conf > KEYPOINT_CONFIDENCE
        and left_eye_conf > KEYPOINT_CONFIDENCE
        and right_eye_conf > KEYPOINT_CONFIDENCE
    ):

        eye_center = (
            left_eye[0] + right_eye[0]
        ) / 2

        eye_distance = abs(
            left_eye[0] - right_eye[0]
        )

        if eye_distance > 5:

            deviation = (
                nose[0] - eye_center
            ) / eye_distance

    # Classify direction
    if deviation is not None:

        if deviation < -HEAD_TURN_THRESHOLD:

            direction = "LEFT"

        elif deviation > HEAD_TURN_THRESHOLD:

            direction = "RIGHT"

        else:

            direction = "FRONT"

    if not LEFT_POINTS_TO_IMAGE_LEFT:

        if direction == "LEFT":
            direction = "RIGHT"

        elif direction == "RIGHT":
            direction = "LEFT"

    return direction, deviation


# ==========================================
# 5. VIDEO ANALYSIS
# ==========================================

def analyse_video(input_path, output_path):

    model = load_model()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open the uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Unable to create the output video."
        )

    # Store each student's turning information
    student_states = {}

    flagged_students = set()

    frame_number = 0

    progress = st.progress(0)

    status_text = st.empty()

    # ======================================
    # PROCESS EACH FRAME
    # ======================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_time = frame_number / fps

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )

        result = results[0]

        visible_ids = set()

        # ==================================
        # GET STUDENT DETECTIONS
        # ==================================

        if (
            result.boxes is not None
            and result.keypoints is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
            )

            keypoints = result.keypoints.xy.cpu().numpy()

            if result.keypoints.conf is not None:

                keypoint_conf = (
                    result.keypoints.conf.cpu().numpy()
                )

            else:

                keypoint_conf = None

            for i, student_id in enumerate(track_ids):

                student_id = int(student_id)

                visible_ids.add(student_id)

                x1, y1, x2, y2 = boxes[i]

                x1 = int(max(0, x1))
                y1 = int(max(0, y1))
                x2 = int(min(width - 1, x2))
                y2 = int(min(height - 1, y2))

                points = keypoints[i]

                # ==================================
                # CALCULATE HEAD DIRECTION
                # ==================================

                if keypoint_conf is not None:

                    confidence = keypoint_conf[i]

                    direction, deviation = (
                        calculate_head_direction(
                            points,
                            confidence
                        )
                    )

                else:

                    direction = "UNCERTAIN"
                    deviation = None

                # ==================================
                # CREATE TRACK STATE
                # ==================================

                if student_id not in student_states:

                    student_states[student_id] = {
                        "direction": "UNCERTAIN",
                        "start_time": None,
                        "turn_duration": 0,
                        "flagged": False,
                        "last_seen": current_time
                    }

                state = student_states[student_id]

                state["last_seen"] = current_time

                # ==================================
                # CHECK SUSTAINED HEAD TURNING
                # ==================================

                if direction in ["LEFT", "RIGHT"]:

                    if state["direction"] != direction:

                        state["direction"] = direction

                        state["start_time"] = current_time

                        state["turn_duration"] = 0

                    else:

                        if state["start_time"] is not None:

                            state["turn_duration"] = (
                                current_time - state["start_time"]
                            )

                    if (
                        state["turn_duration"]
                        >= MIN_TURN_DURATION
                    ):

                        state["flagged"] = True

                        flagged_students.add(student_id)

                else:

                    state["direction"] = direction

                    state["start_time"] = None

                    state["turn_duration"] = 0

                    state["flagged"] = False

                # ==================================
                # DRAW STUDENT BOUNDING BOX
                # ==================================

                if state["flagged"]:

                    box_color = (0, 0, 255)

                    label = (
                        f"ID {student_id}: "
                        f"HEAD TURN - REVIEW REQUIRED"
                    )

                elif direction in ["LEFT", "RIGHT"]:

                    box_color = (0, 255, 255)

                    label = (
                        f"ID {student_id}: "
                        f"TURNING {direction}"
                    )

                elif direction == "FRONT":

                    box_color = (0, 255, 0)

                    label = f"ID {student_id}: FRONT"

                else:

                    box_color = (255, 165, 0)

                    label = (
                        f"ID {student_id}: "
                        f"POSE UNCERTAIN"
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3 if state["flagged"] else 2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                # Display turning duration
                if direction in ["LEFT", "RIGHT"]:

                    duration_text = (
                        f"Turn duration: "
                        f"{state['turn_duration']:.1f}s"
                    )

                    cv2.putText(
                        frame,
                        duration_text,
                        (x1, min(height - 10, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2
                    )

        # ======================================
        # DISPLAY GENERAL WARNING
        # ======================================

        currently_flagged = [
            sid
            for sid, state in student_states.items()
            if (
                state["flagged"]
                and sid in visible_ids
            )
        ]

        if currently_flagged:

            warning = (
                "POTENTIAL CONCERN - "
                "INVIGILATOR REVIEW REQUIRED"
            )

            cv2.putText(
                frame,
                warning,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ======================================
        # WRITE PROCESSED FRAME
        # ======================================

        writer.write(frame)

        frame_number += 1

        # Update progress
        if total_frames > 0:

            progress_value = min(
                frame_number / total_frames,
                1.0
            )

            progress.progress(progress_value)

            status_text.write(
                f"Processing frame {frame_number} "
                f"of {total_frames}"
            )

    # ======================================
    # RELEASE VIDEO RESOURCES
    # ======================================

    cap.release()

    writer.release()

    progress.empty()

    status_text.empty()

    return sorted(flagged_students)


# ==========================================
# 6. VIDEO UPLOAD INTERFACE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Examination Video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ==========================================
# 7. ANALYSE UPLOADED VIDEO
# ==========================================

if uploaded_file is not None:

    st.success("Video uploaded successfully.")

    st.subheader("Original Examination Video")

    st.video(uploaded_file)

    if st.button(
        "Analyse Video",
        type="primary"
    ):

        input_path = None

        try:

            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    uploaded_file.name
                )[1]
            ) as input_file:

                input_file.write(
                    uploaded_file.getbuffer()
                )

                input_path = input_file.name

            # Create output path
            output_path = os.path.join(
                tempfile.gettempdir(),
                "exam_monitoring_output.mp4"
            )

            # Run video analysis
            with st.spinner(
                "Analysing the examination video..."
            ):

                flagged_ids = analyse_video(
                    input_path,
                    output_path
                )

            st.success(
                "Video analysis completed!"
            )

            # ==================================
            # DISPLAY RESULTS
            # ==================================

            st.subheader("Analysis Results")

            if flagged_ids:

                st.error(
                    "Students requiring review: "
                    + ", ".join(
                        map(str, flagged_ids)
                    )
                )

                st.write(
                    "The listed student IDs had at least "
                    "one sustained head-turning observation."
                )

            else:

                st.info(
                    "No sustained head-turning observations "
                    "were detected with the current settings."
                )

            # ==================================
            # DISPLAY OUTPUT VIDEO
            # ==================================

            st.subheader(
                "Processed Examination Video"
            )

            st.video(output_path)

            # ==================================
            # DOWNLOAD OUTPUT VIDEO
            # ==================================

            with open(output_path, "rb") as video_file:

                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="exam_monitoring_output.mp4",
                    mime="video/mp4"
                )

        except Exception as error:

            st.error(
                f"Video processing failed: {error}"
            )

        finally:

            if (
                input_path is not None
                and os.path.exists(input_path)
            ):

                os.remove(input_path)
