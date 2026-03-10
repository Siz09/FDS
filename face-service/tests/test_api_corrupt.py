"""Tests for corrupt-image handling in /embed-face (Day 17 — failure hardening)."""
import io
import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.io_image import is_valid_image
from app.main import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_jpeg_bytes() -> bytes:
    """Encode a small blank BGR image as JPEG bytes."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _corrupt_bytes() -> bytes:
    """Return bytes that are not a valid image."""
    return b"this is not an image \x00\x01\x02\x03"


def _truncated_jpeg_bytes() -> bytes:
    """Return a JPEG that starts with valid header but is cut short."""
    full = _valid_jpeg_bytes()
    return full[: len(full) // 2]  # slice off second half


def _corrupt_png_bytes() -> bytes:
    """Valid PNG signature but corrupt IHDR — PIL rejects it."""
    png_sig = b"\x89PNG\r\n\x1a\n"
    fake_ihdr = struct.pack(">I", 13) + b"IHDR" + b"\xff" * 13 + b"\x00" * 4
    return png_sig + fake_ihdr


# ── Unit tests: is_valid_image ────────────────────────────────────────────────

def test_is_valid_image_accepts_valid_jpeg():
    assert is_valid_image(_valid_jpeg_bytes()) is True


def test_is_valid_image_rejects_random_bytes():
    assert is_valid_image(_corrupt_bytes()) is False


def test_is_valid_image_rejects_truncated_jpeg():
    assert is_valid_image(_truncated_jpeg_bytes()) is False


def test_is_valid_image_rejects_corrupt_png():
    assert is_valid_image(_corrupt_png_bytes()) is False


def test_is_valid_image_rejects_empty_bytes():
    assert is_valid_image(b"") is False


# ── API tests: /embed-face ────────────────────────────────────────────────────

def test_embed_face_corrupt_bytes_returns_422():
    response = client.post(
        "/embed-face",
        files={"image": ("bad.jpg", io.BytesIO(_corrupt_bytes()), "image/jpeg")},
    )
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower()


def test_embed_face_truncated_jpeg_returns_422():
    response = client.post(
        "/embed-face",
        files={"image": ("truncated.jpg", io.BytesIO(_truncated_jpeg_bytes()), "image/jpeg")},
    )
    assert response.status_code == 422


def test_embed_face_valid_blank_image_still_200():
    """Ensure validation doesn't break the normal (no-face) path."""
    response = client.post(
        "/embed-face",
        files={"image": ("blank.jpg", io.BytesIO(_valid_jpeg_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["faces"] == []


def test_embed_face_empty_body_returns_422():
    response = client.post(
        "/embed-face",
        files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code == 422
