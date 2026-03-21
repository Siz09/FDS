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

from app.detector_mediapipe import detect_faces
from app.embedder import FaceRecognitionEmbedder
from app.io_image import load_image, bgr_to_rgb

def test_max_accuracy(img_name):
    img_path = FDS_ROOT / "data" / "mixed" / img_name
    img_bgr = load_image(img_path)
    if img_bgr is None: return
    rgb = bgr_to_rgb(img_bgr)
    
    # 1. Detect
    start = time.time()
    boxes = detect_faces(rgb, min_detection_confidence=0.3)
    print(f"Ensemble found {len(boxes)} faces in {time.time()-start:.2f}s")
    
    # 2. Embed
    embedder = FaceRecognitionEmbedder(model="large")
    known_path = FDS_ROOT / "known" / "person1.npy"
    known_embedding = np.load(known_path)
    
    for i, box in enumerate(boxes[:5]): # Test first 5 faces
        start = time.time()
        # Create a mock image for embedding (normally find_person.py crops it)
        # But we can just test the embedder on the whole image for one face
        # to see the 25 jitters time.
        try:
            # We need the crop for correct embedding
            face_img = rgb[box.y:box.y+box.h, box.x:box.x+box.w]
            if face_img.size == 0: continue
            
            _ = embedder.compute_embedding(face_img, num_jitters=25)
            print(f"Face {i} embedding (25 jitters) took {time.time()-start:.2f}s")
        except Exception as e:
            print(f"Embedding failed: {e}")

if __name__ == "__main__":
    test_max_accuracy("IMG_7411.JPG")
