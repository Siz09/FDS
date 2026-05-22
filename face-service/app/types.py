"""Data types for the Smart Gallery face pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FaceBox:
    """Bounding box for a detected face in pixel coordinates.

    Keypoint order matches MediaPipe blaze_face_full_range:
      0: eye_left  (rightmost in model → smallest x in image)
      1: eye_right
      2: nose
      3: mouth_left
      4: mouth_right
    """

    x: int  # left
    y: int  # top
    w: int  # width
    h: int  # height
    eye_left: Optional[tuple[int, int]] = None    # keypoints[0]
    eye_right: Optional[tuple[int, int]] = None   # keypoints[1]
    nose: Optional[tuple[int, int]] = None        # keypoints[2]
    mouth_left: Optional[tuple[int, int]] = None  # keypoints[3]
    mouth_right: Optional[tuple[int, int]] = None # keypoints[4]

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def landmarks(self) -> list[tuple[int, int]] | None:
        """Returns 5-point landmark list if all are present, else None.

        Order: [eye_left, eye_right, nose, mouth_left, mouth_right].
        Used by align_face_5pt() in sota_onnx.py for ArcFace alignment.
        """
        pts = [self.eye_left, self.eye_right, self.nose, self.mouth_left, self.mouth_right]
        return pts if all(p is not None for p in pts) else None


@dataclass
class FaceResult:
    """A detected face with optional embedding and metadata."""

    bbox: FaceBox
    embedding: Optional["np.ndarray"] = None  # noqa: F821
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.embedding is not None:
            import numpy as np

            arr = self.embedding
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError("embedding must be 1-d numpy array")
            if arr.dtype not in (np.float32, np.float64):
                raise ValueError("embedding must be float32 or float64")


@dataclass
class ImageMatchResult:
    """Result of matching one image against a reference embedding."""

    image_path: str
    matched: bool
    best_similarity: float  # cosine similarity (higher = more similar; was best_distance)
    num_faces: int
    face_boxes: list[FaceBox]
