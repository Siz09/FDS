import urllib.request
import os
from pathlib import Path

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"Done. Size: {os.path.getsize(filename)} bytes")

if __name__ == "__main__":
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Official OpenCV Zoo Models
    download_file(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        models_dir / "yunet.onnx"
    )
    download_file(
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        models_dir / "sface.onnx"
    )
