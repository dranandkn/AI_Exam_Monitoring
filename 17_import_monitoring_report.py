
import sqlite3
import pandas as pd
import os

# File paths
database_path = "database/exam_monitoring.db"
csv_path = "outputs/monitoring_report.csv"
video_path = "outputs/integrated_monitoring.mp4"

# Check whether required files exist
if not os.path.exists(database_path):
    print("Database file not found!")
    exit()

if not os.path.exists(csv_path):
    print("Monitoring CSV file not found!")
    exit()

# Read CSV report
df = pd.read_csv(csv_path)

# Required CSV columns
required_columns = [
    "Timestamp",
    "Student_ID",
    "Head_Position",
    "Head_Movement_Count",
    "Arm_Observation",
    "Phone_Detected",
    "Monitoring_Status"
]

if not all(column in df.columns for column in required_columns):
    print("CSV file does not contain the expected columns.")
    print("Available columns:", list(df.columns))
    exit()

if df.empty:
    print("CSV file contains no monitoring records.")
    exit()

# Connect to database
connection = sqlite3.connect(database_path)
cursor = connection.cursor()

try:
    # Create examination session
    cursor.execute("""
    INSERT INTO sessions (session_name, video_path)
    VALUES (?, ?)
    """, ("Exam Monitoring Session 01", video_path))

    session_id = cursor.lastrowid

    # Replace missing CSV values with None
    df = df.where(pd.notna(df), None)

    # Prepare database records
    records = []

    for _, row in df.iterrows():
        records.append((
            session_id,
            float(row["Timestamp"]),
            int(row["Student_ID"]),
            row["Head_Position"],
            int(row["Head_Movement_Count"]),
            row["Arm_Observation"],
            row["Phone_Detected"],
            row["Monitoring_Status"],
            "PENDING"
        ))

    # Insert monitoring observations
    cursor.executemany("""
    INSERT INTO monitoring_events (
        session_id,
        timestamp,
        student_track_id,
        head_position,
        head_movement_count,
        arm_observation,
        phone_detected,
        monitoring_status,
        review_status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    connection.commit()

    print("CSV DATA IMPORTED SUCCESSFULLY")
    print("--------------------------------")
    print("Session ID:", session_id)
    print("Total records imported:", len(records))
    print("Review status: PENDING")

except Exception as error:
    connection.rollback()
    print("Import failed:", error)

finally:
    connection.close()

print("Database import process completed.")
