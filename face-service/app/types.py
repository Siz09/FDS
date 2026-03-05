"""Data types for the Smart Gallery face pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FaceBox:
    """Bounding box for a detected face in pixel coordinates."""

    x: int  # left
    y: int  # top
    w: int  # width
    h: int  # height

    @property
    def area(self) -> int:
        return self.w * self.h


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
    best_distance: float
    num_faces: int
    face_boxes: list[FaceBox]
