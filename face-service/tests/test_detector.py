"""Unit tests for face detector and related io_image helpers.

Most tests use synthetic numpy images to avoid requiring real photos.
Tests that need a real face image are marked @pytest.mark.integration
and skipped when no fixture image is available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.detector_mediapipe import detect_faces, get_largest_face
from app.io_image import crop_face_region, load_image_from_url
from app.types import FaceBox


# ---------------------------------------------------------------------------
# detect_faces — no-face case
# ---------------------------------------------------------------------------

def test_detect_faces_blank_image_returns_empty():
    """A plain white image contains no faces — detect_faces must return []."""
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)  # white RGB image
    result = detect_faces(blank)
    assert result == []


def test_detect_faces_none_returns_empty():
    """Passing None should return [] without raising."""
    result = detect_faces(None)
    assert result == []


def test_detect_faces_empty_array_returns_empty():
    """Zero-size array should return []."""
    result = detect_faces(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result == []


# ---------------------------------------------------------------------------
# detect_faces — output shape/type validation
# ---------------------------------------------------------------------------

def test_detect_faces_returns_list():
    """Return value is always a list."""
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(blank)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_largest_face
# ---------------------------------------------------------------------------

def test_get_largest_face_empty_returns_none():
    assert get_largest_face([]) is None


def test_get_largest_face_single():
    box = FaceBox(x=10, y=10, w=50, h=50)
    assert get_largest_face([box]) is box


def test_get_largest_face_picks_biggest():
    small = FaceBox(x=0, y=0, w=20, h=20)   # area 400
    big   = FaceBox(x=50, y=50, w=80, h=80)  # area 6400
    assert get_largest_face([small, big]) is big
    assert get_largest_face([big, small]) is big


def test_get_largest_face_equal_sizes():
    """When two boxes have the same area, returns one of them (not None)."""
    a = FaceBox(x=0, y=0, w=30, h=30)
    b = FaceBox(x=5, y=5, w=30, h=30)
    result = get_largest_face([a, b])
    assert result in (a, b)


# ---------------------------------------------------------------------------
# crop_face_region — clamping
# ---------------------------------------------------------------------------

def test_crop_face_region_normal():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = FaceBox(x=20, y=20, w=30, h=30)
    crop = crop_face_region(image, bbox, pad_fraction=0.0)
    assert crop.shape == (30, 30, 3)


def test_crop_face_region_clamped_to_bounds():
    """Bbox that extends past image edge should be clamped, not raise."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox extends well beyond image boundary
    bbox = FaceBox(x=90, y=90, w=50, h=50)
    crop = crop_face_region(image, bbox, pad_fraction=0.0)
    h, w = crop.shape[:2]
    assert h <= 100
    assert w <= 100


def test_crop_face_region_with_padding_clamped():
    """20% padding on a bbox near the edge must not exceed image bounds."""
    image = np.ones((100, 100, 3), dtype=np.uint8)
    bbox = FaceBox(x=0, y=0, w=40, h=40)
    crop = crop_face_region(image, bbox, pad_fraction=0.2)
    h, w = crop.shape[:2]
    assert h <= 100
    assert w <= 100


# ---------------------------------------------------------------------------
# load_image_from_url
# ---------------------------------------------------------------------------

def test_load_image_from_url_invalid_host_returns_none():
    """Unreachable URL must return None, not raise."""
    result = load_image_from_url("http://localhost:19999/nonexistent.jpg", timeout=2)
    assert result is None


def test_load_image_from_url_bad_scheme_returns_none():
    """Unsupported scheme must return None, not raise."""
    result = load_image_from_url("ftp://example.com/image.jpg", timeout=2)
    assert result is None


def test_load_image_from_url_valid_bytes():
    """When urlopen returns valid JPEG bytes, the result is a BGR numpy array."""
    import cv2

    # Create a tiny test image and encode it as JPEG bytes
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    jpeg_bytes = encoded.tobytes()

    class _FakeResponse:
        def read(self):
            return jpeg_bytes
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=_FakeResponse()):
        result = load_image_from_url("http://fake.url/image.jpg")

    assert result is not None
    assert result.shape == (50, 50, 3)


# ---------------------------------------------------------------------------
# Integration: real face image (skipped if fixture not present)
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FACE_IMAGE = FIXTURE_DIR / "face.jpg"


@pytest.mark.integration
@pytest.mark.skipif(not FACE_IMAGE.exists(), reason="No test face fixture at tests/fixtures/face.jpg")
def test_detect_faces_real_image_finds_at_least_one():
    import cv2
    from app.io_image import bgr_to_rgb, load_image

    img_bgr = load_image(FACE_IMAGE)
    assert img_bgr is not None
    img_rgb = bgr_to_rgb(img_bgr)
    boxes = detect_faces(img_rgb)
    assert len(boxes) >= 1
    for box in boxes:
        assert box.w > 0
        assert box.h > 0
        assert box.x >= 0
        assert box.y >= 0
