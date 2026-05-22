"""Image I/O and preprocessing for the face pipeline.

- Load with OpenCV (BGR). Convert to RGB for detection/embedding.
- Crop face region with configurable padding, clamped to image bounds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Add parent for imports when run as script
if __name__ != "__main__":
    pass
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.types import FaceBox


def load_image(path: str | Path) -> np.ndarray | None:
    """Load image with OpenCV. Returns BGR array or None if load fails."""
    path = Path(path)
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray | None:
    """Load image from bytes with OpenCV. Returns BGR array or None if load fails."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for MediaPipe and face_recognition."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def align_face(image: np.ndarray, left_eye: tuple[int, int], right_eye: tuple[int, int]) -> np.ndarray:
    """Rotate image so eyes are level. Returns rotated full image."""
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0,
                  (left_eye[1] + right_eye[1]) / 2.0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_face_region(
    image: np.ndarray,
    bbox: FaceBox,
    pad_fraction: float = 0.2,
) -> np.ndarray:
    """Crop face region with padding, performing 2D alignment if eyes are detected.

    Args:
        image: Full image (BGR or RGB).
        bbox: Face bounding box in pixel coords (x, y, w, h).
        pad_fraction: Fraction of bbox size to add as padding (e.g. 0.2 = 20%).

    Returns:
        Cropped and aligned patch.
    """
    img_to_crop = image
    if bbox.eye_left and bbox.eye_right:
        img_to_crop = align_face(image, bbox.eye_left, bbox.eye_right)

    h_img, w_img = img_to_crop.shape[:2]
    pad_w = max(0, int(bbox.w * pad_fraction))
    pad_h = max(0, int(bbox.h * pad_fraction))
    
    x1 = max(0, bbox.x - pad_w)
    y1 = max(0, bbox.y - pad_h)
    x2 = min(w_img, bbox.x + bbox.w + pad_w)
    y2 = min(h_img, bbox.y + bbox.h + pad_h)
    
    return img_to_crop[y1:y2, x1:x2].copy()


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of image."""
    h, w = image.shape[:2]
    return w, h
