import sys
import os
from pathlib import Path
import cv2
import numpy as np
import face_recognition

# Set paths
FDS_ROOT = Path(r"D:\vscode\photoDEx\FDS")
sys.path.insert(0, str(FDS_ROOT / "face-service"))

from app.detector_mediapipe import detect_faces
from app.io_image import load_image, bgr_to_rgb

def test_image(img_name):
    img_path = FDS_ROOT / "data" / "mixed" / img_name
    img_bgr = load_image(img_path)
    if img_bgr is None:
        print(f"Failed to load {img_name}")
        return
    rgb = bgr_to_rgb(img_bgr)
    
    print(f"Testing {img_name}...")
    
    # Test MediaPipe
    mp_boxes = detect_faces(rgb, min_detection_confidence=0.3)
    print(f"MediaPipe (0.3 conf) found {len(mp_boxes)} faces")
    
    # Test HOG directly
    hog_locations = face_recognition.face_locations(rgb, model="hog")
    print(f"face_recognition HOG found {len(hog_locations)} faces")

if __name__ == "__main__":
    test_image("IMG_7411.JPG")
    test_image("IMG_7419.JPG")
