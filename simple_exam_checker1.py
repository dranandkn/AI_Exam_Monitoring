
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from ultralytics import YOLO


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="AI Exam Monitoring",
    page_icon="🎓",
    layout="wide"
)

st.title("AI-Based Examination Video Monitoring")
st.write("Upload an examination video to identify students repeatedly turning towards nearby candidates.")

st.warning(
    "The system highlights potentially concerning behaviour for invigilator review. "
    "A head turn alone does not establish cheating."
)


# -----------------------------------------
# LOAD YOLO MODEL
# -----------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")


# -----------------------------------------
# VIDEO PROCESSING FUNCTION
# -----------------------------------------

def analyse_video(input_path, output_path):

    model = load_model()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Unable to open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create the output video.")

    # Detection settings
    LOOKING_THRESHOLD = 0.20
    MIN_LOOK_DURATION = 1.5

    # If the direction appears reversed in your camera,
    # change this setting to False.
    LEFT_POINTS_TO_IMAGE_LEFT = True

    # Store information for each tracked student
    student_states = {}

    frame_number = 0
    suspicious_students = set()

    progress = st.progress(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_time = frame_number / fps

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            verbose=False
        )

        result = results[0]

        students = []

        if result.boxes is not None and result.keypoints is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            if result.boxes.id is not None:

                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i, track_id in enumerate(track_ids):

                    x1, y1, x2, y2 = boxes[i]

                    person_width = max(1, x2 - x1)
                    person_height = max(1, y2 - y1)

                    points = keypoints[i]

                    # COCO keypoints:
                    # 0 = Nose
                    # 1 = Left eye
                    # 2 = Right eye

                    nose = points[0]
                    left_eye = points[1]
                    right_eye = points[2]

                    # Estimate head direction
                    eye_center_x = (left_eye[0] + right_eye[0]) / 2

                    eye_distance = abs(
                        left_eye[0] - right_eye[0]
                    )

                    direction = "CENTER"

                    if eye_distance > 2:

                        deviation = (
                            nose[0] - eye_center_x
                        ) / eye_distance

                        if deviation < -LOOKING_THRESHOLD:
                            direction = "LEFT"

                        elif deviation > LOOKING_THRESHOLD:
                            direction = "RIGHT"

                    if not LEFT_POINTS_TO_IMAGE_LEFT:
                        if direction == "LEFT":
                            direction = "RIGHT"
                        elif direction == "RIGHT":
                            direction = "LEFT"

                    students.append({
                        "id": int(track_id),
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                        "width": person_width,
                        "height": person_height,
                        "direction": direction
                    })

        # -----------------------------------------
        # FIND NEIGHBOUR IN HEAD-TURN DIRECTION
        # -----------------------------------------

        for student in students:

            student_id = student["id"]
            direction = student["direction"]

            target_neighbour = None

            if direction in ["LEFT", "RIGHT"]:

                possible_neighbours = []

                for other in students:

                    if other["id"] == student_id:
                        continue

                    vertical_difference = abs(
                        student["center_y"] - other["center_y"]
                    )

                    same_row = vertical_difference <= (
                        0.55 * max(
                            student["height"],
                            other["height"]
                        )
                    )

                    horizontal_gap = abs(
                        student["center_x"] - other["center_x"]
                    )

                    nearby = horizontal_gap <= (
                        2.5 * student["width"]
                    )

                    if direction == "LEFT":
                        correct_side = (
                            other["center_x"] < student["center_x"]
                        )
                    else:
                        correct_side = (
                            other["center_x"] > student["center_x"]
                        )

                    if same_row and nearby and correct_side:

                        possible_neighbours.append(
                            (horizontal_gap, other["id"])
                        )

                if possible_neighbours:

                    possible_neighbours.sort()
                    target_neighbour = possible_neighbours[0][1]

            # -----------------------------------------
            # CHECK HOW LONG STUDENT FACES NEIGHBOUR
            # -----------------------------------------

            if student_id not in student_states:

                student_states[student_id] = {
                    "target": None,
                    "start_time": None
                }

            state = student_states[student_id]

            if target_neighbour is not None:

                if state["target"] != target_neighbour:

                    state["target"] = target_neighbour
                    state["start_time"] = current_time

                duration = current_time - state["start_time"]

                if duration >= MIN_LOOK_DURATION:

                    suspicious_students.add(student_id)

                    x1, y1, x2, y2 = student["box"]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    label = (
                        f"ID {student_id}: REVIEW - "
                        f"LOOKING AT NEIGHBOUR"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(25, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                else:

                    x1, y1, x2, y2 = student["box"]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID {student_id}: Checking...",
                        (int(x1), max(25, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

            else:

                state["target"] = None
                state["start_time"] = None

                x1, y1, x2, y2 = student["box"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Student ID: {student_id}",
                    (int(x1), max(25, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Add a warning banner if any student is highlighted
        if any(
            student_states[sid]["target"] is not None
            and (
                current_time - student_states[sid]["start_time"]
            ) >= MIN_LOOK_DURATION
            for sid in student_states
        ):

            cv2.putText(
                frame,
                "POTENTIAL CHEATING - REVIEW REQUIRED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        writer.write(frame)

        frame_number += 1

        if total_frames > 0:
            progress.progress(
                min(frame_number / total_frames, 1.0)
            )

    cap.release()
    writer.release()

    progress.empty()

    return sorted(suspicious_students)


# -----------------------------------------
# VIDEO UPLOAD SECTION
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload Examination Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    st.success("Video uploaded successfully.")

    st.video(uploaded_file)

    if st.button("Analyse Video", type="primary"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as input_file:

            input_file.write(uploaded_file.getbuffer())
            input_path = input_file.name

        output_path = os.path.join(
            tempfile.gettempdir(),
            "exam_monitoring_output.mp4"
        )

        try:

            with st.spinner("Analysing examination video..."):

                flagged_ids = analyse_video(
                    input_path,
                    output_path
                )

            st.success("Video analysis completed!")

            st.subheader("Analysis Results")

            if flagged_ids:

                st.error(
                    "Students requiring review: "
                    + ", ".join(map(str, flagged_ids))
                )

                st.write(
                    "These student IDs were highlighted in the video."
                )

            else:

                st.info(
                    "No sustained neighbour-facing observations "
                    "were detected using the current settings."
                )

            st.subheader("Processed Examination Video")

            st.video(output_path)

            with open(output_path, "rb") as video_file:

                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="exam_monitoring_output.mp4",
                    mime="video/mp4"
                )

        except Exception as error:

            st.error(f"Video processing failed: {error}")

        finally:

            if os.path.exists(input_path):
                os.remove(input_path)
