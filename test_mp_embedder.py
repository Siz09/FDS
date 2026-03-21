import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2

# Set paths
FDS_ROOT = Path(r"D:\vscode\photoDEx\FDS")
sys.path.insert(0, str(FDS_ROOT / "face-service"))

from mediapipe.tasks.python import base
from mediapipe.tasks.python.vision import face_embedder
from app.io_image import load_image, bgr_to_rgb
from app.detector_mediapipe import detect_faces

def test_mediapipe_embedder(img_name):
    img_path = FDS_ROOT / "data" / "mixed" / img_name
    img_bgr = load_image(img_path)
    if img_bgr is None: return
    rgb = bgr_to_rgb(img_bgr)
    
    # 1. Download model if missing (usually handled by Tasks API or manual)
    # The model URL is usually: 
    # https://storage.googleapis.com/mediapipe-models/face_embedder/face_be_embedder/float16/1/face_be_embedder.tflite
    model_path = FDS_ROOT / "models" / "face_embedder.tflite"
    if not model_path.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_embedder/face_be_embedder/float16/1/face_be_embedder.tflite"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, model_path)

    # 2. Setup Embedder
    options = face_embedder.FaceEmbedderOptions(
        base_options=base.BaseOptions(model_asset_path=str(model_path))
    )
    
    with face_embedder.FaceEmbedder.create_from_options(options) as embedder:
        # Detect faces first
        boxes = detect_faces(rgb)
        print(f"Detected {len(boxes)} faces.")
        
        from mediapipe import Image, ImageFormat
        mp_image = Image(ImageFormat.SRGB, rgb)
        
        for i, box in enumerate(boxes[:3]):
            start = time.time()
            # MediaPipe tasks can take a region of interest (ROI)
            # but usually they embed the most prominent face or we can provide a crop
            # Actually, FaceEmbedder.embed(mp_image) returns embeddings for all faces!
            result = embedder.embed(mp_image)
            if result.embeddings:
                print(f"Face {i} embedded in {time.time()-start:.4f}s")
                print(f"Embeddings count: {len(result.embeddings)}")
                # result.embeddings is a list of Embedding objects
                vec = result.embeddings[0].float_vector
                print(f"Vector size: {len(vec)}")

if __name__ == "__main__":
    test_mediapipe_embedder("IMG_7411.JPG")
