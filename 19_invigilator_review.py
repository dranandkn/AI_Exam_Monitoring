
import sqlite3

database_path = "database/exam_monitoring.db"

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

# Display observations requiring review
cursor.execute("""
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
WHERE monitoring_status = 'REVIEW'
AND review_status = 'PENDING'
ORDER BY timestamp
""")

alerts = cursor.fetchall()

if not alerts:
    print("No pending review alerts found.")

else:
    print("\nPENDING INVIGILATOR REVIEW")
    print("=" * 60)

    for alert in alerts:
        print("\nEvent ID:", alert[0])
        print("Timestamp:", alert[1], "seconds")
        print("Student Track ID:", alert[2])
        print("Head Position:", alert[3])
        print("Head Movement Count:", alert[4])
        print("Arm Observation:", alert[5])
        print("Phone Detected:", alert[6])
        print("AI Monitoring Status:", alert[7])
        print("Review Status:", alert[8])
        print("-" * 60)

    # Select an event for review
    try:
        event_id = int(input("\nEnter Event ID to review: "))

        cursor.execute("""
        SELECT event_id
        FROM monitoring_events
        WHERE event_id = ?
        AND monitoring_status = 'REVIEW'
        AND review_status = 'PENDING'
        """, (event_id,))

        selected_event = cursor.fetchone()

        if selected_event is None:
            print("Invalid Event ID or event already reviewed.")

        else:
            print("\nSelect Review Status:")
            print("1. REVIEWED")
            print("2. DISMISSED")
            print("3. Keep PENDING")

            choice = input("Enter your choice: ").strip()

            status_options = {
                "1": "REVIEWED",
                "2": "DISMISSED",
                "3": "PENDING"
            }

            if choice not in status_options:
                print("Invalid choice.")

            else:
                new_status = status_options[choice]

                comments = input(
                    "Enter reviewer comments: "
                ).strip()

                cursor.execute("""
                UPDATE monitoring_events
                SET review_status = ?,
                    reviewer_comments = ?
                WHERE event_id = ?
                """, (new_status, comments, event_id))

                connection.commit()

                print("\nREVIEW UPDATED SUCCESSFULLY")
                print("Event ID:", event_id)
                print("Review Status:", new_status)

    except ValueError:
        print("Please enter a valid numeric Event ID.")

connection.close()
