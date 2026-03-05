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
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _get_model_path() -> Path:
    """Return path to face detector .tflite; download if missing."""
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / "blaze_face_short_range.tflite"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_DETECTOR_MODEL_URL, path)
    return path


def detect_faces(
    rgb_image: np.ndarray,
    min_detection_confidence: float = 0.5,
    model_selection: int = 1,
    max_faces: int = 10,
) -> List[FaceBox]:
    """Detect faces in an RGB image using MediaPipe Face Detector (Tasks API).

    Args:
        rgb_image: Image in RGB (H, W, 3), uint8.
        min_detection_confidence: Minimum confidence [0, 1] for a detection.
        model_selection: Ignored (Tasks API uses short-range model only for now).
        max_faces: Maximum number of faces to return (by confidence order).

    Returns:
        List of FaceBox in pixel coords (x, y, w, h). Empty if no faces.
    """
    if rgb_image is None or rgb_image.size == 0:
        return []

    model_path = _get_model_path()
    base_options = base_options_module.BaseOptions(
        model_asset_path=str(model_path)
    )
    options = FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_detection_confidence,
    )
    detector = FaceDetector.create_from_options(options)
    try:
        # MediaPipe Image from numpy RGB (contiguous uint8).
        if not rgb_image.flags.c_contiguous:
            rgb_image = np.ascontiguousarray(rgb_image)
        mp_image = Image(ImageFormat.SRGB, rgb_image)
        result = detector.detect(mp_image)
    finally:
        detector.close()

    if not result.detections:
        return []

    boxes: List[FaceBox] = []
    for det in result.detections[:max_faces]:
        bbox = det.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height
        if w > 0 and h > 0:
            boxes.append(FaceBox(x=x, y=y, w=w, h=h))
    return boxes


def get_largest_face(boxes: List[FaceBox]) -> FaceBox | None:
    """Return the face box with largest area, or None if empty."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b.area)
