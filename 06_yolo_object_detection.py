
from ultralytics import YOLO
import cv2

# Load the pretrained YOLO11 model
print("Loading YOLO11 model...")

model = YOLO("yolo11n.pt")

# Give the input image path
image_path = "images/captured_image.jpg"

# Perform object detection
print("Detecting objects...")

results = model.predict(
    source=image_path,
    conf=0.35,
    save=True
)

# Display the detection results
for result in results:
    print("\nDetected Objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"Object: {object_name}, "
            f"Confidence: {confidence:.2f}"
        )

print("\nDetection completed!")
