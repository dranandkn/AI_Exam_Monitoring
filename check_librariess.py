
import cv2
import numpy
import torch
import ultralytics
import pandas
import streamlit
import matplotlib

print("\n================================")
print(" AI EXAM MONITORING ENVIRONMENT")
print("================================\n")

print("OpenCV      :", cv2.__version__)
print("NumPy       :", numpy.__version__)
print("PyTorch     :", torch.__version__)
print("Ultralytics :", ultralytics.__version__)
print("Pandas      :", pandas.__version__)
print("Streamlit   :", streamlit.__version__)
print("Matplotlib  :", matplotlib.__version__)

print("\nCUDA Available:", torch.cuda.is_available())

print("\nALL LIBRARIES IMPORTED SUCCESSFULLY!")
