"""Matching logic: compare face embeddings with a tolerance (Euclidean distance).

- An image is a match if ANY face in it is within tolerance of the reference.
- best_distance = min of distances from reference to each face in the image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

if __name__ != "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedder import euclidean_distance
from app.types import FaceBox, FaceResult, ImageMatchResult


def match_image(
    reference_embedding: np.ndarray,
    face_results: List[FaceResult],
    tolerance: float,
) -> ImageMatchResult:
    """Determine if an image matches the reference (any face within tolerance).

    Args:
        reference_embedding: 1-d reference vector (float32/64).
        face_results: List of FaceResult with embeddings set.
        tolerance: Max Euclidean distance to consider a match (lower = stricter).

    Returns:
        ImageMatchResult with matched=True if any face distance <= tolerance.
    """
    if not face_results:
        return ImageMatchResult(
            image_path="",
            matched=False,
            best_distance=float("inf"),
            num_faces=0,
            face_boxes=[],
        )

    distances: List[float] = []
    for fr in face_results:
        if fr.embedding is not None:
            d = euclidean_distance(reference_embedding, fr.embedding)
            distances.append(d)

    if not distances:
        return ImageMatchResult(
            image_path="",
            matched=False,
            best_distance=float("inf"),
            num_faces=len(face_results),
            face_boxes=[fr.bbox for fr in face_results],
        )

    best_distance = min(distances)
    matched = best_distance <= tolerance
    return ImageMatchResult(
        image_path="",
        matched=matched,
        best_distance=best_distance,
        num_faces=len(face_results),
        face_boxes=[fr.bbox for fr in face_results],
    )


def is_match(best_distance: float, tolerance: float) -> bool:
    """Return True if best_distance <= tolerance."""
    return best_distance <= tolerance
