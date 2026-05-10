"""Face detection using MediaPipe Face Detector (Tasks API).

- Input: RGB image.
- Output: list of FaceBox (x, y, w, h in pixel coords).
- Uses blaze_face_short_range model (downloaded on first use).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy as np

if __name__ != "__main__":
    pass
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

from app.types import FaceBox

# Short-range BlazeFace model (same as MediaPipe samples).
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)


def _get_model_path() -> Path:
    """Return path to face detector .tflite; download if missing."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / "blaze_face_full_range.tflite"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_DETECTOR_MODEL_URL, path)
    return path


_detector: "FaceDetector | None" = None  # set in lifespan only, never at import time


def _get_detector() -> "FaceDetector":
    """Return the process-level MediaPipe detector singleton (confidence=0.3)."""
    global _detector
    if _detector is None:
        model_path = _get_model_path()
        options = FaceDetectorOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.3,
        )
        _detector = FaceDetector.create_from_options(options)
    return _detector


def detect_faces(
    rgb_image: np.ndarray,
    model_selection: int = 1,   # kept for call-site compatibility, unused
    max_faces: int = 10,
) -> List[FaceBox]:
    """Detect faces in an RGB image using MediaPipe Face Detector (Tasks API).

    Args:
        rgb_image: Image in RGB (H, W, 3), uint8.
        model_selection: Ignored (Tasks API uses short-range model only for now).
        max_faces: Maximum number of faces to return (by confidence order).

    Returns:
        List of FaceBox in pixel coords (x, y, w, h). Empty if no faces.
    """
    if rgb_image is None or rgb_image.size == 0:
        return []

    detector = _get_detector()
    # MediaPipe Image from numpy RGB (contiguous uint8).
    if not rgb_image.flags.c_contiguous:
        rgb_image = np.ascontiguousarray(rgb_image)
    mp_image = Image(ImageFormat.SRGB, rgb_image)
    result = detector.detect(mp_image)

    img_h, img_w = rgb_image.shape[:2]

    # MediaPipe detections only — HOG fallback removed (produced noisy crops).
    mp_boxes: List[FaceBox] = []
    if result.detections:
        for det in result.detections[:max_faces]:
            bbox = det.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            if w > 0 and h > 0:
                eye_left = None
                eye_right = None
                if det.keypoints and len(det.keypoints) >= 2:
                    kp0 = det.keypoints[0]
                    kp1 = det.keypoints[1]
                    pt0 = (int(kp0.x * img_w), int(kp0.y * img_h))
                    pt1 = (int(kp1.x * img_w), int(kp1.y * img_h))
                    # Assign left/right based on x-coordinate in image
                    if pt0[0] < pt1[0]:
                        eye_left, eye_right = pt0, pt1
                    else:
                        eye_left, eye_right = pt1, pt0

                mp_boxes.append(FaceBox(x=x, y=y, w=w, h=h, eye_left=eye_left, eye_right=eye_right))

    if not mp_boxes:
        return []

    # 3. Merge overlapping boxes (IoU logic) to avoid duplicates
    def get_iou(box1, box2):
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.w, box2.x + box2.w)
        y2 = min(box1.y + box1.h, box2.y + box2.h)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.w * box1.h
        area2 = box2.w * box2.h
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    final_boxes: List[FaceBox] = []
    for box in mp_boxes:
        is_duplicate = False
        for existing in final_boxes:
            if get_iou(box, existing) > 0.5: # 50% overlap threshold
                is_duplicate = True
                break
        if not is_duplicate:
            final_boxes.append(box)

    return final_boxes[:max_faces]


def get_largest_face(boxes: List[FaceBox]) -> FaceBox | None:
    """Return the face box with largest area, or None if empty."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b.area)
