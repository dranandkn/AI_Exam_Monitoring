
import cv2

# Open recorded video
video = cv2.VideoCapture(
    "videos/recorded_video.mp4"
)

if not video.isOpened():
    print("Unable to open video")
    exit()

# Get video information
fps = video.get(cv2.CAP_PROP_FPS)

total_frames = video.get(
    cv2.CAP_PROP_FRAME_COUNT
)

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("Video Information")
print("-----------------")

print("FPS:", fps)
print("Total Frames:", int(total_frames))
print("Width:", int(width))
print("Height:", int(height))

# Read and display frames
while True:

    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow("Recorded Examination Video", frame)

    # Press Q to stop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Video processing completed!")
