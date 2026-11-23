
import cv2

print("OpenCV successfully installed!")

print("OpenCV Version:", cv2.__version__)

image = cv2.imread("test.jpg")

if image is None:
    print("Image not available yet.")
else:
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
