
import sqlite3
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

connection = sqlite3.connect(
    "database/exam_monitoring.db"
)

query = """
SELECT
    student_track_id AS Student_Track_ID,
    COUNT(*) AS Total_Observations,
    SUM(CASE WHEN monitoring_status = 'REVIEW'
        THEN 1 ELSE 0 END) AS Review_Observations,
    SUM(CASE WHEN phone_detected = 'YES'
        THEN 1 ELSE 0 END) AS Phone_Detections,
    SUM(CASE WHEN review_status = 'PENDING'
        AND monitoring_status = 'REVIEW'
        THEN 1 ELSE 0 END) AS Pending_Reviews,
    SUM(CASE WHEN review_status = 'REVIEWED'
        THEN 1 ELSE 0 END) AS Reviewed_Observations,
    SUM(CASE WHEN review_status = 'DISMISSED'
        THEN 1 ELSE 0 END) AS Dismissed_Observations
FROM monitoring_events
GROUP BY student_track_id
ORDER BY student_track_id
"""

df = pd.read_sql_query(query, connection)

connection.close()

output_path = "outputs/student_analytics.csv"

df.to_csv(output_path, index=False)

print("ANALYTICS EXPORTED SUCCESSFULLY")
print("Output file:", output_path)
print("\nStudent-wise report:")
print(df)
