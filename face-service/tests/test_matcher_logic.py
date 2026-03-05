"""Tests for matcher logic (threshold, multi-face, no face)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.matcher import match_image, is_match
from app.types import FaceResult, FaceBox, ImageMatchResult


def _box(x: int = 0, y: int = 0, w: int = 64, h: int = 64) -> FaceBox:
    return FaceBox(x=x, y=y, w=w, h=h)


def _embedding(value: float = 0.0, dim: int = 128) -> np.ndarray:
    return np.full(dim, value, dtype=np.float64)


def test_is_match() -> None:
    assert is_match(0.3, 0.6) is True
    assert is_match(0.6, 0.6) is True
    assert is_match(0.61, 0.6) is False
    assert is_match(0.0, 0.6) is True


def test_match_image_no_faces() -> None:
    ref = _embedding(0.0)
    result = match_image(ref, [], tolerance=0.6)
    assert result.matched is False
    assert result.num_faces == 0
    assert result.best_distance == float("inf")
    assert result.face_boxes == []


def test_match_image_one_face_under_tolerance() -> None:
    ref = _embedding(0.0)
    # Same embedding -> distance 0
    face = FaceResult(bbox=_box(), embedding=_embedding(0.0))
    result = match_image(ref, [face], tolerance=0.6)
    assert result.matched is True
    assert result.best_distance == pytest.approx(0.0, abs=1e-6)
    assert result.num_faces == 1


def test_match_image_one_face_over_tolerance() -> None:
    ref = _embedding(0.0)
    # Far embedding
    far = _embedding(10.0)
    face = FaceResult(bbox=_box(), embedding=far)
    result = match_image(ref, [face], tolerance=0.6)
    assert result.matched is False
    assert result.best_distance > 0.6
    assert result.num_faces == 1


def test_match_image_any_face_matches() -> None:
    """Image is match if ANY face is within tolerance."""
    ref = _embedding(0.0)
    far_face = FaceResult(bbox=_box(0, 0, 32, 32), embedding=_embedding(10.0))
    near_face = FaceResult(bbox=_box(64, 0, 32, 32), embedding=_embedding(0.0))
    result = match_image(ref, [far_face, near_face], tolerance=0.6)
    assert result.matched is True
    assert result.best_distance == pytest.approx(0.0, abs=1e-6)
    assert result.num_faces == 2


def test_match_image_best_distance_is_min() -> None:
    ref = _embedding(0.0)
    d1 = np.linalg.norm(_embedding(0.1) - ref)
    d2 = np.linalg.norm(_embedding(0.05) - ref)
    face1 = FaceResult(bbox=_box(), embedding=_embedding(0.1))
    face2 = FaceResult(bbox=_box(64, 0), embedding=_embedding(0.05))
    result = match_image(ref, [face1, face2], tolerance=1.0)
    assert result.best_distance == pytest.approx(min(d1, d2), abs=1e-5)
