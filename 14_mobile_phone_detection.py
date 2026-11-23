
from ultralytics import YOLO
import cv2
import math

# Load YOLO11 object detection model
print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

# Input video
video_path = "videos/recorded_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 20.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "outputs/mobile_phone_detection.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# COCO class IDs
PERSON_CLASS = 0
PHONE_CLASS = 67

total_phone_detections = 0
frames_with_phone = 0

print("Mobile phone detection started...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track people and mobile phones
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS, PHONE_CLASS],
        conf=0.35,
        verbose=False
    )

    result = results[0]

    persons = []
    phones = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(classes)

        # Separate people and phones
        for box, class_id, confidence, track_id in zip(
            boxes, classes, confidences, track_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            if class_id == PERSON_CLASS:

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id
                })

            elif class_id == PHONE_CLASS:

                phones.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })

    # Draw person bounding boxes
    for person in persons:

        x1, y1, x2, y2 = person["box"]
        track_id = person["track_id"]

        label = (
            f"Person ID {track_id}"
            if track_id is not None
            else "Person"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 150, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 150, 0),
            2
        )

    # Process detected mobile phones
    if phones:
        frames_with_phone += 1

    for phone in phones:

        x1, y1, x2, y2 = phone["box"]
        confidence = phone["confidence"]

        total_phone_detections += 1

        # Find a person whose bounding box contains
        # the centre of the detected phone
        phone_center_x = (x1 + x2) // 2
        phone_center_y = (y1 + y2) // 2

        associated_person = None

        for person in persons:

            px1, py1, px2, py2 = person["box"]

            if (
                px1 <= phone_center_x <= px2
                and py1 <= phone_center_y <= py2
            ):

                associated_person = person
                break

        # Create phone label
        if (
            associated_person is not None
            and associated_person["track_id"] is not None
        ):

            student_label = (
                f"Student ID {associated_person['track_id']}"
            )

        elif associated_person is not None:

            student_label = "Person detected"

        else:

            student_label = "UNASSIGNED"

        label = (
            f"CELL PHONE {confidence:.2f} | "
            f"{student_label}"
        )

        # Draw phone bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # Display phone detection count
    cv2.putText(
        frame,
        f"Phones detected in frame: {len(phones)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Display video
    cv2.imshow(
        "Examination - Mobile Phone Detection",
        frame
    )

    # Save processed video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nMobile phone detection completed!")

print("Total phone detection instances:", total_phone_detections)
print("Frames containing phone detections:", frames_with_phone)

print("Output saved at:", output_path)
