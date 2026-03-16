"""Tests for API key authentication middleware."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_no_key_returns_200():
    """/health is exempt from auth."""
    response = client.get("/health")
    assert response.status_code == 200


def test_stats_no_key_returns_200():
    """/stats is exempt from auth."""
    response = client.get("/stats")
    assert response.status_code == 200


def test_embed_no_key_returns_403(monkeypatch):
    """POST /embed-face with no key → 403."""
    monkeypatch.setenv("FACE_SERVICE_API_KEY", "test-secret-key-32chars-padded-xx")
    response = client.post("/embed-face", files={"image": ("f.jpg", b"fake", "image/jpeg")})
    assert response.status_code == 403


def test_embed_wrong_key_returns_403(monkeypatch):
    """POST /embed-face with wrong key → 403."""
    monkeypatch.setenv("FACE_SERVICE_API_KEY", "test-secret-key-32chars-padded-xx")
    response = client.post(
        "/embed-face",
        files={"image": ("f.jpg", b"fake", "image/jpeg")},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


def test_embed_correct_key_not_403(monkeypatch):
    """POST /embed-face with correct key → not 403 (may be 200 or 422)."""
    monkeypatch.setenv("FACE_SERVICE_API_KEY", "test-secret-key-32chars-padded-xx")
    response = client.post(
        "/embed-face",
        files={"image": ("f.jpg", b"fake", "image/jpeg")},
        headers={"X-API-Key": "test-secret-key-32chars-padded-xx"},
    )
    assert response.status_code in (200, 422)


def test_detect_no_key_returns_403(monkeypatch):
    """POST /detect-face with no key → 403."""
    monkeypatch.setenv("FACE_SERVICE_API_KEY", "test-secret-key-32chars-padded-xx")
    response = client.post("/detect-face", files={"image": ("f.jpg", b"fake", "image/jpeg")})
    assert response.status_code == 403


def test_match_no_key_returns_403(monkeypatch):
    """POST /match-face with no key → 403."""
    monkeypatch.setenv("FACE_SERVICE_API_KEY", "test-secret-key-32chars-padded-xx")
    response = client.post(
        "/match-face",
        files={
            "reference": ("ref.jpg", b"fake", "image/jpeg"),
            "target": ("target.jpg", b"fake", "image/jpeg"),
        },
    )
    assert response.status_code == 403
