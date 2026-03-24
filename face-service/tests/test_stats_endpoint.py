"""Integration tests for the /stats HTTP endpoint.

Auth note: FACE_SERVICE_API_KEY is not set in the test environment, so
APIKeyMiddleware skips auth (self.api_key is None). Tests use TestClient(app)
with no extra headers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import app.stats as stats_module
from app.main import app
from app.stats import StatsCollector


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stats():
    """Replace singleton before each test.
    main.py uses `import app.stats as _stats_mod` so it dereferences
    _stats_mod.stats on each call — module-level replacement works correctly.
    """
    stats_module.stats = StatsCollector()
    yield


def test_stats_returns_200(client):
    resp = client.get("/stats")
    assert resp.status_code == 200


def test_stats_response_schema(client):
    data = client.get("/stats").json()
    assert "uptime_s" in data
    assert "requests" in data
    assert "faces" in data
    assert "latency_ms" in data
    assert "system" in data
    assert "pid" in data["system"]
    assert "python_version" in data["system"]


def test_stats_health_not_tracked(client):
    client.get("/health")
    assert "health" not in client.get("/stats").json()["requests"]
