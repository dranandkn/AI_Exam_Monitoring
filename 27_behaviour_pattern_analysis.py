
import os
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

OUTPUT_DIRECTORY = "outputs"

NEIGHBOUR_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "neighbour_looking_report.csv"
)

PHONE_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "mobile_phone_usage_report.csv"
)

FREQUENT_LOOKING_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "frequent_looking_report.csv"
)

INTERACTION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "student_interaction_report.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIRECTORY,
    "behaviour_review_summary.csv"
)

EVENT_OUTPUT = os.path.join(
    OUTPUT_DIRECTORY,
    "consolidated_behaviour_events.csv"
)


# ---------------------------------------------------------
# 2. REVIEW PRIORITY WEIGHTS
# ---------------------------------------------------------

WEIGHTS = {

    "neighbour_looking": 3,

    "phone_observation": 4,

    "frequent_looking": 2,

    "student_interaction": 3

}


# Review category thresholds
LOW_SCORE = 1

MODERATE_SCORE = 4

HIGH_SCORE = 8


# ---------------------------------------------------------
# 3. STUDENT-WISE STORAGE
# ---------------------------------------------------------

student_data = defaultdict(lambda: {

    "neighbour_count": 0,

    "phone_count": 0,

    "frequent_looking_count": 0,

    "interaction_count": 0,

    "total_neighbour_duration": 0.0,

    "total_phone_duration": 0.0,

    "total_frequent_duration": 0.0,

    "total_interaction_duration": 0.0,

    "review_score": 0,

    "observations": set()

})


consolidated_events = []


# ---------------------------------------------------------
# 4. NORMALIZE STUDENT ID
# ---------------------------------------------------------

def normalize_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    try:

        numeric_value = float(value)

        if numeric_value.is_integer():

            return str(int(numeric_value))

    except ValueError:

        pass

    return value


# ---------------------------------------------------------
# 5. SAFE NUMERIC CONVERSION
# ---------------------------------------------------------

def safe_number(value):

    try:

        if pd.isna(value):
            return 0.0

        return float(value)

    except (ValueError, TypeError):

        return 0.0


# ---------------------------------------------------------
# 6. FUNCTION TO RECORD AN OBSERVATION
# ---------------------------------------------------------

def record_observation(
    student_id,
    observation_type,
    event,
    duration,
    score,
    start_time,
    end_time,
    related_student=None
):

    if student_id is None:
        return

    data = student_data[student_id]

    # Update observation count and duration
    if observation_type == "neighbour_looking":

        data["neighbour_count"] += 1

        data["total_neighbour_duration"] += duration

        data["observations"].add(
            "Sustained neighbour-facing"
        )

    elif observation_type == "phone_observation":

        data["phone_count"] += 1

        data["total_phone_duration"] += duration

        data["observations"].add(
            "Phone near wrist"
        )

    elif observation_type == "frequent_looking":

        data["frequent_looking_count"] += 1

        data["total_frequent_duration"] += duration

        data["observations"].add(
            "Frequent head-direction changes"
        )

    elif observation_type == "student_interaction":

        data["interaction_count"] += 1

        data["total_interaction_duration"] += duration

        data["observations"].add(
            "Possible hand-to-hand interaction"
        )

    # Add weighted review points
    data["review_score"] += score

    # Store consolidated event
    consolidated_events.append({

        "Student_ID": student_id,

        "Related_Student_ID": (
            related_student if related_student else ""
        ),

        "Observation_Type": observation_type,

        "Start_Time": start_time,

        "End_Time": end_time,

        "Duration_Seconds": round(duration, 2),

        "Review_Points": score

    })


# ---------------------------------------------------------
# 7. READ NEIGHBOUR-LOOKING REPORT
# ---------------------------------------------------------

print("\nReading neighbour-looking report...")

if os.path.exists(NEIGHBOUR_FILE):

    df = pd.read_csv(NEIGHBOUR_FILE)

    for _, row in df.iterrows():

        student_id = normalize_id(
            row.get("Student_ID")
        )

        duration = safe_number(
            row.get("duration_seconds")
        )

        record_observation(

            student_id,

            "neighbour_looking",

            "SUSTAINED NEIGHBOUR-FACING",

            duration,

            WEIGHTS["neighbour_looking"],

            row.get("start_time", ""),

            row.get("end_time", ""),

            normalize_id(
                row.get("Neighbour_Track_ID")
            )

        )

    print("Neighbour-looking report processed.")

else:

    print("WARNING: Neighbour-looking report not found.")


# ---------------------------------------------------------
# 8. READ MOBILE PHONE REPORT
# ---------------------------------------------------------

print("\nReading mobile phone report...")

if os.path.exists(PHONE_FILE):

    df = pd.read_csv(PHONE_FILE)

    for _, row in df.iterrows():

        student_id = normalize_id(
            row.get("Student_ID")
        )

        duration = safe_number(
            row.get("duration_seconds")
        )

        record_observation(

            student_id,

            "phone_observation",

            "PHONE NEAR STUDENT WRIST",

            duration,

            WEIGHTS["phone_observation"],

            row.get("start_time", ""),

            row.get("end_time", "")

        )

    print("Mobile phone report processed.")

else:

    print("WARNING: Mobile phone report not found.")


# ---------------------------------------------------------
# 9. READ FREQUENT LOOKING REPORT
# ---------------------------------------------------------

print("\nReading frequent-looking report...")

if os.path.exists(FREQUENT_LOOKING_FILE):

    df = pd.read_csv(FREQUENT_LOOKING_FILE)

    for _, row in df.iterrows():

        student_id = normalize_id(
            row.get("Student_ID")
        )

        duration = safe_number(
            row.get("duration_seconds")
        )

        record_observation(

            student_id,

            "frequent_looking",

            "FREQUENT HEAD DIRECTION CHANGES",

            duration,

            WEIGHTS["frequent_looking"],

            row.get("start_time", ""),

            row.get("end_time", "")

        )

    print("Frequent-looking report processed.")

else:

    print("WARNING: Frequent-looking report not found.")


# ---------------------------------------------------------
# 10. READ STUDENT INTERACTION REPORT
# ---------------------------------------------------------

print("\nReading student interaction report...")

if os.path.exists(INTERACTION_FILE):

    df = pd.read_csv(INTERACTION_FILE)

    for _, row in df.iterrows():

        student_a = normalize_id(
            row.get("Student_ID_1")
        )

        student_b = normalize_id(
            row.get("Student_ID_2")
        )

        duration = safe_number(
            row.get("duration_seconds")
        )

        start_time = row.get("start_time", "")

        end_time = row.get("end_time", "")

        # Record interaction for first student
        record_observation(

            student_a,

            "student_interaction",

            "POSSIBLE HAND-TO-HAND INTERACTION",

            duration,

            WEIGHTS["student_interaction"],

            start_time,

            end_time,

            student_b

        )

        # Record interaction for second student
        record_observation(

            student_b,

            "student_interaction",

            "POSSIBLE HAND-TO-HAND INTERACTION",

            duration,

            WEIGHTS["student_interaction"],

            start_time,

            end_time,

            student_a

        )

    print("Student interaction report processed.")

else:

    print("WARNING: Student interaction report not found.")


# ---------------------------------------------------------
# 11. DETERMINE REVIEW CATEGORY
# ---------------------------------------------------------

def get_review_category(score):

    if score == 0:

        return "NO RECORDED OBSERVATION"

    elif score < MODERATE_SCORE:

        return "LOW REVIEW PRIORITY"

    elif score < HIGH_SCORE:

        return "MODERATE REVIEW PRIORITY"

    else:

        return "HIGH REVIEW PRIORITY"


# ---------------------------------------------------------
# 12. GENERATE STUDENT-WISE SUMMARY
# ---------------------------------------------------------

summary_records = []


for student_id, data in student_data.items():

    score = data["review_score"]

    category = get_review_category(score)

    observation_types = ", ".join(
        sorted(data["observations"])
    )

    summary_records.append({

        "Student_ID": student_id,

        "Neighbour_Looking_Count": (
            data["neighbour_count"]
        ),

        "Phone_Observation_Count": (
            data["phone_count"]
        ),

        "Frequent_Looking_Count": (
            data["frequent_looking_count"]
        ),

        "Interaction_Count": (
            data["interaction_count"]
        ),

        "Neighbour_Duration_Seconds": round(
            data["total_neighbour_duration"], 2
        ),

        "Phone_Duration_Seconds": round(
            data["total_phone_duration"], 2
        ),

        "Frequent_Looking_Duration_Seconds": round(
            data["total_frequent_duration"], 2
        ),

        "Interaction_Duration_Seconds": round(
            data["total_interaction_duration"], 2
        ),

        "Review_Priority_Score": score,

        "Review_Category": category,

        "Observed_Behaviours": observation_types

    })


# ---------------------------------------------------------
# 13. CREATE DATAFRAME
# ---------------------------------------------------------

summary_df = pd.DataFrame(summary_records)


# Sort students by review priority score
if not summary_df.empty:

    summary_df = summary_df.sort_values(

        by="Review_Priority_Score",

        ascending=False

    )


# ---------------------------------------------------------
# 14. SAVE STUDENT SUMMARY
# ---------------------------------------------------------

summary_df.to_csv(

    SUMMARY_OUTPUT,

    index=False,

    encoding="utf-8"

)


# ---------------------------------------------------------
# 15. SAVE CONSOLIDATED EVENT REPORT
# ---------------------------------------------------------

event_columns = [

    "Student_ID",

    "Related_Student_ID",

    "Observation_Type",

    "Start_Time",

    "End_Time",

    "Duration_Seconds",

    "Review_Points"

]


event_df = pd.DataFrame(

    consolidated_events,

    columns=event_columns

)


event_df.to_csv(

    EVENT_OUTPUT,

    index=False,

    encoding="utf-8"

)


# ---------------------------------------------------------
# 16. DISPLAY FINAL REPORT
# ---------------------------------------------------------

print("\n======================================")

print("INTEGRATED BEHAVIOUR ANALYSIS COMPLETED")

print("======================================")

print("\nTotal Students with Recorded Observations:")

print(len(summary_df))


print("\nStudent-wise Review Summary:")

if not summary_df.empty:

    print(

        summary_df[
            [
                "Student_ID",

                "Neighbour_Looking_Count",

                "Phone_Observation_Count",

                "Frequent_Looking_Count",

                "Interaction_Count",

                "Review_Priority_Score",

                "Review_Category"

            ]

        ].to_string(index=False)

    )

else:

    print("No observations were found.")


print("\nOutput Files:")

print("Student Summary:", SUMMARY_OUTPUT)

print("Consolidated Events:", EVENT_OUTPUT)


print("\nReview priority analysis completed.")

print("All observations require human verification.")
