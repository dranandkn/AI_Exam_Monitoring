
import streamlit as st
import sqlite3
import pandas as pd
import os

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Exam Monitoring Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# DATABASE CONNECTION
# --------------------------------

DATABASE_PATH = "database/exam_monitoring.db"

st.title("🎓 AI-Based Examination Monitoring System")

st.caption(
    "Real-Time Student Monitoring and Invigilator Review Dashboard"
)

st.divider()

if not os.path.exists(DATABASE_PATH):
    st.error("Monitoring database not found!")
    st.stop()

connection = sqlite3.connect(DATABASE_PATH)

# --------------------------------
# GET EXAMINATION SESSIONS
# --------------------------------

sessions = pd.read_sql_query(
    "SELECT * FROM sessions ORDER BY session_id DESC",
    connection
)

if sessions.empty:
    st.warning("No examination sessions available.")
    connection.close()
    st.stop()

session_ids = sessions["session_id"].tolist()

selected_session = st.selectbox(
    "Select Examination Session",
    session_ids,
    format_func=lambda x: (
        f"Session {x} - "
        f"{sessions.loc[sessions['session_id'] == x, 'session_name'].iloc[0]}"
    )
)

# --------------------------------
# RETRIEVE SUMMARY STATISTICS
# --------------------------------

query = """
SELECT
    COUNT(*) AS total_observations,
    COUNT(DISTINCT student_track_id) AS total_students,

    SUM(CASE
        WHEN monitoring_status = 'REVIEW'
        THEN 1 ELSE 0 END) AS review_alerts,

    SUM(CASE
        WHEN phone_detected = 'YES'
        THEN 1 ELSE 0 END) AS phone_detections,

    SUM(CASE
        WHEN monitoring_status = 'REVIEW'
        AND review_status = 'PENDING'
        THEN 1 ELSE 0 END) AS pending_reviews

FROM monitoring_events
WHERE session_id = ?
"""

summary = pd.read_sql_query(
    query,
    connection,
    params=(selected_session,)
).iloc[0]

total_students = int(summary["total_students"] or 0)
total_observations = int(summary["total_observations"] or 0)
review_alerts = int(summary["review_alerts"] or 0)
phone_detections = int(summary["phone_detections"] or 0)
pending_reviews = int(summary["pending_reviews"] or 0)

# --------------------------------
# DISPLAY DASHBOARD METRICS
# --------------------------------

st.subheader("Examination Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Student Tracks",
    total_students
)

col2.metric(
    "Total Observations",
    total_observations
)

col3.metric(
    "Review Alerts",
    review_alerts
)

col4.metric(
    "Pending Reviews",
    pending_reviews
)

st.divider()

# --------------------------------
# STUDENT-WISE ANALYTICS
# --------------------------------

st.subheader("Student-Wise Monitoring Analytics")

student_query = """
SELECT
    student_track_id,
    COUNT(*) AS total_observations,

    SUM(CASE
        WHEN monitoring_status = 'REVIEW'
        THEN 1 ELSE 0 END) AS review_observations,

    SUM(CASE
        WHEN phone_detected = 'YES'
        THEN 1 ELSE 0 END) AS phone_detections

FROM monitoring_events
WHERE session_id = ?
GROUP BY student_track_id
ORDER BY student_track_id
"""

student_data = pd.read_sql_query(
    student_query,
    connection,
    params=(selected_session,)
)

if not student_data.empty:

    left, right = st.columns(2)

    with left:
        st.write("Total Observations by Student")

        chart_data = student_data.set_index(
            "student_track_id"
        )[["total_observations"]]

        st.bar_chart(chart_data)

    with right:
        st.write("Review Observations by Student")

        review_chart = student_data.set_index(
            "student_track_id"
        )[["review_observations"]]

        st.bar_chart(review_chart)

    st.write("Student Analytics Table")

    st.dataframe(
        student_data,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No student analytics available.")

st.divider()

# --------------------------------
# RECENT MONITORING OBSERVATIONS
# --------------------------------

st.subheader("Recent Monitoring Observations")

events_query = """
SELECT
    event_id,
    timestamp,
    student_track_id,
    head_position,
    head_movement_count,
    arm_observation,
    phone_detected,
    monitoring_status,
    review_status

FROM monitoring_events

WHERE session_id = ?

ORDER BY timestamp DESC

LIMIT 100
"""

events_data = pd.read_sql_query(
    events_query,
    connection,
    params=(selected_session,)
)

if not events_data.empty:

    st.dataframe(
        events_data,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No monitoring observations available.")

connection.close()

st.divider()

st.caption(
    "AI-generated observations are intended for invigilator review. "
    "They do not independently establish examination misconduct."
)

# --------------------------------
# INVIGILATOR REVIEW MANAGEMENT
# --------------------------------

st.divider()

st.subheader("Invigilator Review Management")

# Open a separate database connection
review_connection = sqlite3.connect(DATABASE_PATH)

pending_query = """
SELECT
    event_id,
    timestamp,
    student_track_id,
    head_position,
    head_movement_count,
    arm_observation,
    phone_detected,
    monitoring_status,
    review_status
FROM monitoring_events
WHERE session_id = ?
AND monitoring_status = 'REVIEW'
AND review_status = 'PENDING'
ORDER BY timestamp
"""

pending_data = pd.read_sql_query(
    pending_query,
    review_connection,
    params=(selected_session,)
)

if pending_data.empty:

    st.success("There are no pending review observations.")

else:

    st.write(
        f"Pending observations available: {len(pending_data)}"
    )

    # Select an observation
    event_options = pending_data["event_id"].tolist()

    selected_event = st.selectbox(
        "Select an Event ID to review",
        event_options
    )

    event_details = pending_data[
        pending_data["event_id"] == selected_event
    ].iloc[0]

    # Display selected event details
    st.markdown("### Selected Observation Details")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Student Track ID",
        int(event_details["student_track_id"])
    )

    col2.metric(
        "Timestamp",
        f"{event_details['timestamp']:.2f} sec"
    )

    col3.metric(
        "Head Movement Count",
        int(event_details["head_movement_count"])
    )

    st.write("Head Position:", event_details["head_position"])
    st.write("Arm Observation:", event_details["arm_observation"])
    st.write("Phone Detected:", event_details["phone_detected"])
    st.write("AI Monitoring Status:", event_details["monitoring_status"])

    st.divider()

    # Invigilator review form
    st.markdown("### Invigilator Decision")

    with st.form("invigilator_review_form"):

        review_decision = st.selectbox(
            "Select Review Status",
            ["REVIEWED", "DISMISSED"]
        )

        reviewer_comments = st.text_area(
            "Reviewer Comments",
            placeholder="Enter your observations after checking the video..."
        )

        submit_review = st.form_submit_button(
            "Save Review Decision"
        )

        if submit_review:

            update_cursor = review_connection.cursor()

            update_cursor.execute("""
            UPDATE monitoring_events
            SET review_status = ?,
                reviewer_comments = ?
            WHERE event_id = ?
            AND session_id = ?
            AND review_status = 'PENDING'
            """, (
                review_decision,
                reviewer_comments,
                int(selected_event),
                int(selected_session)
            ))

            review_connection.commit()

            if update_cursor.rowcount == 1:

                st.success(
                    f"Event {selected_event} updated successfully."
                )

                st.info(
                    f"New Review Status: {review_decision}"
                )

                st.rerun()

            else:
                st.warning(
                    "The selected event was already updated. "
                    "Please refresh the dashboard."
                )

review_connection.close()

# --------------------------------
# EXAMINATION VIDEO PLAYBACK
# --------------------------------

st.divider()

st.subheader("Examination Video Playback")

# Retrieve video path for selected session
video_connection = sqlite3.connect(DATABASE_PATH)

video_query = """
SELECT video_path
FROM sessions
WHERE session_id = ?
"""

video_result = pd.read_sql_query(
    video_query,
    video_connection,
    params=(selected_session,)
)

video_connection.close()

if video_result.empty:

    st.warning("No video information found for this session.")

else:

    video_path = video_result.iloc[0]["video_path"]

    if not video_path or not os.path.exists(video_path):

        st.error(
            "Examination video file not found: "
            + str(video_path)
        )

    else:

        st.write("Examination Video")

        st.video(video_path)

        st.divider()

        # Retrieve monitoring events
        event_connection = sqlite3.connect(DATABASE_PATH)

        event_query = """
        SELECT
            event_id,
            timestamp,
            student_track_id,
            head_position,
            head_movement_count,
            arm_observation,
            phone_detected,
            monitoring_status,
            review_status
        FROM monitoring_events
        WHERE session_id = ?
        ORDER BY timestamp
        """

        video_events = pd.read_sql_query(
            event_query,
            event_connection,
            params=(selected_session,)
        )

        event_connection.close()

        if not video_events.empty:

            st.markdown("### Examine a Specific Monitoring Event")

            selected_video_event = st.selectbox(
                "Select Event ID",
                video_events["event_id"].tolist(),
                key="video_event_selector"
            )

            selected_details = video_events[
                video_events["event_id"] == selected_video_event
            ].iloc[0]

            event_timestamp = float(
                selected_details["timestamp"]
            )

            st.markdown("### Selected Event Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Event ID",
                int(selected_details["event_id"])
            )

            col2.metric(
                "Student Track ID",
                int(selected_details["student_track_id"])
            )

            col3.metric(
                "Event Timestamp",
                f"{event_timestamp:.2f} sec"
            )

            st.write(
                "Head Position:",
                selected_details["head_position"]
            )

            st.write(
                "Head Movement Count:",
                int(selected_details["head_movement_count"])
            )

            st.write(
                "Arm Observation:",
                selected_details["arm_observation"]
            )

            st.write(
                "Phone Detected:",
                selected_details["phone_detected"]
            )

            st.write(
                "Monitoring Status:",
                selected_details["monitoring_status"]
            )

            st.write(
                "Invigilator Review Status:",
                selected_details["review_status"]
            )

            st.divider()

            st.markdown("### Play Video Near Selected Event")

            st.caption(
                "The video player will start near the selected "
                "event timestamp. Check the surrounding footage "
                "to understand the observation."
            )

            st.video(
                video_path,
                start_time=int(event_timestamp)
            )

        else:

            st.info("No monitoring events available for this session.")

# --------------------------------
# DASHBOARD FILTERS AND SEARCH
# --------------------------------

st.divider()

st.subheader("🔎 Advanced Monitoring Search")

# Connect to database
filter_connection = sqlite3.connect(DATABASE_PATH)

filter_query = """
SELECT
    event_id,
    timestamp,
    student_track_id,
    head_position,
    head_movement_count,
    arm_observation,
    phone_detected,
    monitoring_status,
    review_status,
    reviewer_comments
FROM monitoring_events
WHERE session_id = ?
ORDER BY timestamp DESC
"""

all_events = pd.read_sql_query(
    filter_query,
    filter_connection,
    params=(selected_session,)
)

filter_connection.close()

if all_events.empty:

    st.info("No monitoring records available.")

else:

    # --------------------------------
    # FILTER CONTROLS
    # --------------------------------

    st.markdown("### Filter Monitoring Records")

    col1, col2 = st.columns(2)

    with col1:

        student_search = st.text_input(
            "Search Student Track ID",
            placeholder="Enter a student tracking ID"
        )

        monitoring_filter = st.selectbox(
            "Monitoring Status",
            ["ALL", "REVIEW", "OBSERVATION"]
        )

        phone_filter = st.selectbox(
            "Phone Detection",
            ["ALL", "YES", "NO"]
        )

    with col2:

        review_filter = st.selectbox(
            "Invigilator Review Status",
            ["ALL", "PENDING", "REVIEWED", "DISMISSED"]
        )

        event_search = st.text_input(
            "Search Event ID",
            placeholder="Enter an event ID"
        )

    # --------------------------------
    # APPLY FILTERS
    # --------------------------------

    filtered_events = all_events.copy()

    if student_search.strip():

        if student_search.strip().isdigit():

            student_id = int(student_search.strip())

            filtered_events = filtered_events[
                filtered_events["student_track_id"] == student_id
            ]

        else:

            st.warning(
                "Student Track ID must be a numeric value."
            )

    if event_search.strip():

        if event_search.strip().isdigit():

            event_id = int(event_search.strip())

            filtered_events = filtered_events[
                filtered_events["event_id"] == event_id
            ]

        else:

            st.warning(
                "Event ID must be a numeric value."
            )

    if monitoring_filter != "ALL":

        filtered_events = filtered_events[
            filtered_events["monitoring_status"]
            == monitoring_filter
        ]

    if phone_filter != "ALL":

        filtered_events = filtered_events[
            filtered_events["phone_detected"]
            == phone_filter
        ]

    if review_filter != "ALL":

        filtered_events = filtered_events[
            filtered_events["review_status"]
            == review_filter
        ]

    # --------------------------------
    # DISPLAY FILTERED RESULTS
    # --------------------------------

    st.divider()

    st.markdown("### Filtered Monitoring Results")

    st.metric(
        "Matching Records",
        len(filtered_events)
    )

    if not filtered_events.empty:

        st.dataframe(
            filtered_events,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------
        # EXPORT FILTERED RESULTS
        # --------------------------------

        csv_data = filtered_events.to_csv(index=False)

        st.download_button(
            label="Download Filtered Report",
            data=csv_data,
            file_name="filtered_monitoring_report.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No records match the selected filters."
        )

    # --------------------------------
    # STUDENT-WISE EVENT ANALYSIS
    # --------------------------------

    st.divider()

    st.markdown("### Student-Wise Event Analysis")

    if not filtered_events.empty:

        student_summary = (
            filtered_events
            .groupby("student_track_id")
            .agg(
                Total_Observations=("event_id", "count"),
                Review_Observations=(
                    "monitoring_status",
                    lambda x: (x == "REVIEW").sum()
                ),
                Phone_Detections=(
                    "phone_detected",
                    lambda x: (x == "YES").sum()
                ),
                Pending_Reviews=(
                    "review_status",
                    lambda x: (x == "PENDING").sum()
                )
            )
            .reset_index()
        )

        student_summary = student_summary.rename(
            columns={
                "student_track_id": "Student_Track_ID"
            }
        )

        st.dataframe(
            student_summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Student Observation Chart")

        chart_data = student_summary.set_index(
            "Student_Track_ID"
        )[["Total_Observations", "Review_Observations"]]

        st.bar_chart(chart_data)

    else:

        st.info(
            "Student-wise analysis is unavailable "
            "because no records match the filters."
        )
