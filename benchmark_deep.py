import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
import face_recognition

# Set paths
FDS_ROOT = Path(r"D:\vscode\photoDEx\FDS")
sys.path.insert(0, str(FDS_ROOT / "face-service"))

from app.io_image import load_image, bgr_to_rgb

def benchmark_detectors(img_name):
    img_path = FDS_ROOT / "data" / "mixed" / img_name
    img_bgr = load_image(img_path)
    if img_bgr is None: return
    rgb = bgr_to_rgb(img_bgr)
    
    print(f"\n--- Benchmarking {img_name} ---")
    
    # HOG
    start = time.time()
    hog_locs = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1)
    print(f"HOG (upsample 1) found {len(hog_locs)} faces in {time.time()-start:.2f}s")

    # CNN (May fail if dlib not compiled with CUDA, will use CPU)
    try:
        start = time.time()
        cnn_locs = face_recognition.face_locations(rgb, model="cnn")
        print(f"CNN found {len(cnn_locs)} faces in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"CNN failed: {e}")

if __name__ == "__main__":
    benchmark_detectors("IMG_7411.JPG")
