
import sqlite3

database_path = "database/exam_monitoring.db"

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

print("\n==========================================")
print("       AI EXAM MONITORING REPORT")
print("==========================================")

# 1. Total examination sessions
cursor.execute("""
SELECT COUNT(*) FROM sessions
""")
total_sessions = cursor.fetchone()[0]

# 2. Total unique student tracking IDs
cursor.execute("""
SELECT COUNT(DISTINCT student_track_id)
FROM monitoring_events
""")
total_students = cursor.fetchone()[0]

# 3. Total monitoring observations
cursor.execute("""
SELECT COUNT(*) FROM monitoring_events
""")
total_events = cursor.fetchone()[0]

# 4. Observations requiring review
cursor.execute("""
SELECT COUNT(*)
FROM monitoring_events
WHERE monitoring_status = 'REVIEW'
""")
review_alerts = cursor.fetchone()[0]

# 5. Total phone detection observations
cursor.execute("""
SELECT COUNT(*)
FROM monitoring_events
WHERE phone_detected = 'YES'
""")
phone_detections = cursor.fetchone()[0]

# 6. Pending reviews
cursor.execute("""
SELECT COUNT(*)
FROM monitoring_events
WHERE review_status = 'PENDING'
AND monitoring_status = 'REVIEW'
""")
pending_reviews = cursor.fetchone()[0]

# 7. Reviewed observations
cursor.execute("""
SELECT COUNT(*)
FROM monitoring_events
WHERE review_status = 'REVIEWED'
""")
reviewed_events = cursor.fetchone()[0]

# 8. Dismissed observations
cursor.execute("""
SELECT COUNT(*)
FROM monitoring_events
WHERE review_status = 'DISMISSED'
""")
dismissed_events = cursor.fetchone()[0]

# Display summary
print("\nGENERAL STATISTICS")
print("------------------------------------------")
print("Total Sessions             :", total_sessions)
print("Unique Student Tracks      :", total_students)
print("Total Monitoring Records   :", total_events)
print("Observations for Review    :", review_alerts)
print("Phone Detection Records    :", phone_detections)
print("Pending Reviews            :", pending_reviews)
print("Reviewed Records           :", reviewed_events)
print("Dismissed Records          :", dismissed_events)

# Student-wise statistics
print("\nSTUDENT-WISE MONITORING SUMMARY")
print("------------------------------------------")

cursor.execute("""
SELECT
    student_track_id,
    COUNT(*) AS total_observations,
    SUM(CASE WHEN monitoring_status = 'REVIEW'
        THEN 1 ELSE 0 END) AS review_observations,
    SUM(CASE WHEN phone_detected = 'YES'
        THEN 1 ELSE 0 END) AS phone_observations
FROM monitoring_events
GROUP BY student_track_id
ORDER BY student_track_id
""")

student_records = cursor.fetchall()

print(
    f"{'Track ID':<12}"
    f"{'Total':<12}"
    f"{'Review':<12}"
    f"{'Phone':<12}"
)

print("-" * 48)

for record in student_records:
    print(
        f"{record[0]:<12}"
        f"{record[1]:<12}"
        f"{record[2]:<12}"
        f"{record[3]:<12}"
    )

connection.close()

print("\n==========================================")
print("       REPORT GENERATION COMPLETED")
print("==========================================")
