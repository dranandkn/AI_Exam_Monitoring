
import sqlite3
import os

# Database folder and file
os.makedirs("database", exist_ok=True)

database_path = "database/exam_monitoring.db"

# Connect to SQLite database
connection = sqlite3.connect(database_path)

cursor = connection.cursor()

# Table 1: Examination Sessions
cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT,
    video_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Table 2: Monitoring Events
cursor.execute("""
CREATE TABLE IF NOT EXISTS monitoring_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp REAL,
    student_track_id INTEGER,
    head_position TEXT,
    head_movement_count INTEGER,
    arm_observation TEXT,
    phone_detected TEXT,
    monitoring_status TEXT,
    review_status TEXT DEFAULT 'PENDING',
    reviewer_comments TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
""")

connection.commit()
connection.close()

print("DATABASE CREATED SUCCESSFULLY")
print("Database location:", database_path)
print("Tables created:")
print("1. sessions")
print("2. monitoring_events")
