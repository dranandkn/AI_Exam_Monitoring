
import cv2
import numpy
import torch
import ultralytics
import pandas
import streamlit
import matplotlib

print("ALL LIBRARIES INSTALLED SUCCESSFULLY")

print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Pandas:", pandas.__version__)
print("Streamlit:", streamlit.__version__)
print("Matplotlib:", matplotlib.__version__)

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")