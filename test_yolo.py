
from ultralytics import YOLO

print("YOLO successfully installed!")

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")

print("Model task:", model.task)
