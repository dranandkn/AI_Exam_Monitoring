
import sqlite3

connection = sqlite3.connect(
    "database/exam_monitoring.db"
)

cursor = connection.cursor()

# Display examination sessions
print("\nEXAMINATION SESSIONS")
print("--------------------")

cursor.execute("SELECT * FROM sessions")

for row in cursor.fetchall():
    print(row)

# Display first 10 monitoring records
print("\nMONITORING EVENTS")
print("-----------------")

cursor.execute("""
SELECT
    event_id,
    timestamp,
    student_track_id,
    head_position,
    phone_detected,
    monitoring_status,
    review_status
FROM monitoring_events
LIMIT 10
""")

for row in cursor.fetchall():
    print(row)

# Display total records
cursor.execute("""
SELECT COUNT(*) FROM monitoring_events
""")

total = cursor.fetchone()[0]

print("\nTotal monitoring records:", total)

connection.close()
