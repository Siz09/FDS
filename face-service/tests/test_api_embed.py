"""Tests for the POST /embed-face endpoint."""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FIXTURE_FACE = "tests/fixtures/face.jpg"


@pytest.mark.integration
def test_embed_face_returns_faces_key():
    with open(FIXTURE_FACE, "rb") as f:
        response = client.post("/embed-face", files={"image": ("face.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    assert "faces" in response.json()


@pytest.mark.integration
def test_embed_face_embeddings_are_128d():
    with open(FIXTURE_FACE, "rb") as f:
        response = client.post("/embed-face", files={"image": ("face.jpg", f, "image/jpeg")})
    data = response.json()
    assert len(data["faces"]) >= 1
    for face in data["faces"]:
        assert len(face["embedding"]) == 128


@pytest.mark.integration
def test_embed_face_box_fields_present():
    with open(FIXTURE_FACE, "rb") as f:
        response = client.post("/embed-face", files={"image": ("face.jpg", f, "image/jpeg")})
    data = response.json()
    for face in data["faces"]:
        assert all(k in face["box"] for k in ("x", "y", "w", "h"))


def test_embed_face_blank_image_returns_empty_faces():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    response = client.post(
        "/embed-face",
        files={"image": ("blank.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["faces"] == []
