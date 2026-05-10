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


def crop_face_region(
    image: np.ndarray,
    bbox: FaceBox,
    pad_fraction: float = 0.2,
) -> np.ndarray:
    """Crop face region with padding, clamped to image boundaries.

    Args:
        image: Full image (BGR or RGB).
        bbox: Face bounding box in pixel coords (x, y, w, h).
        pad_fraction: Fraction of bbox size to add as padding (e.g. 0.2 = 20%).

    Returns:
        Cropped patch (same color order as input).
    """
    h_img, w_img = image.shape[:2]

    if bbox.eye_left and bbox.eye_right:
        dy = bbox.eye_right[1] - bbox.eye_left[1]
        dx = bbox.eye_right[0] - bbox.eye_left[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Only align if tilted > 3 degrees
        if abs(angle) > 3.0:
            # Safe padding for rotation (to avoid black corners)
            diag = np.sqrt(bbox.w**2 + bbox.h**2)
            safe_pad_w = int((diag - bbox.w) / 2 + bbox.w * pad_fraction)
            safe_pad_h = int((diag - bbox.h) / 2 + bbox.h * pad_fraction)
            
            x1_safe = max(0, bbox.x - safe_pad_w)
            y1_safe = max(0, bbox.y - safe_pad_h)
            x2_safe = min(w_img, bbox.x + bbox.w + safe_pad_w)
            y2_safe = min(h_img, bbox.y + bbox.h + safe_pad_h)
            
            safe_crop = image[y1_safe:y2_safe, x1_safe:x2_safe]
            
            # Center of the face relative to the safe crop
            center_x = (bbox.x + bbox.w / 2.0) - x1_safe
            center_y = (bbox.y + bbox.h / 2.0) - y1_safe
            
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_crop = cv2.warpAffine(
                safe_crop, M, (safe_crop.shape[1], safe_crop.shape[0]), 
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            
            # Now extract the final padded face from the rotated safe crop
            pad_w = int(bbox.w * pad_fraction)
            pad_h = int(bbox.h * pad_fraction)
            
            rx1 = max(0, int(center_x - bbox.w / 2.0 - pad_w))
            ry1 = max(0, int(center_y - bbox.h / 2.0 - pad_h))
            rx2 = min(rotated_crop.shape[1], int(center_x + bbox.w / 2.0 + pad_w))
            ry2 = min(rotated_crop.shape[0], int(center_y + bbox.h / 2.0 + pad_h))
            
            if rx2 > rx1 and ry2 > ry1:
                return rotated_crop[ry1:ry2, rx1:rx2].copy()

    # Standard unrotated crop (fallback)
    pad_w = max(0, int(bbox.w * pad_fraction))
    pad_h = max(0, int(bbox.h * pad_fraction))
    x1 = max(0, bbox.x - pad_w)
    y1 = max(0, bbox.y - pad_h)
    x2 = min(w_img, bbox.x + bbox.w + pad_w)
    y2 = min(h_img, bbox.y + bbox.h + pad_h)
    return image[y1:y2, x1:x2].copy()


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of image."""
    h, w = image.shape[:2]
    return w, h
