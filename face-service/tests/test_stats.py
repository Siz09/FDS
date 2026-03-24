"""Tests for StatsCollector — thread-safe in-process metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stats import StatsCollector


def test_initial_snapshot_is_empty():
    sc = StatsCollector()
    snap = sc.get_snapshot()
    assert snap["requests"] == {}
    assert snap["faces"]["detected"] == 0
    assert snap["faces"]["no_face_images"] == 0
    assert snap["latency_ms"] == {}
    assert snap["uptime_s"] >= 0
    assert "system" in snap


def test_record_increments_total():
    sc = StatsCollector()
    sc.record("embed_face", latency_ms=100.0, num_faces=2, error=False)
    snap = sc.get_snapshot()
    assert snap["requests"]["embed_face"]["total"] == 1
    assert snap["requests"]["embed_face"]["errors"] == 0


def test_record_error_increments_error_count():
    sc = StatsCollector()
    sc.record("embed_face", latency_ms=50.0, num_faces=0, error=True)
    snap = sc.get_snapshot()
    assert snap["requests"]["embed_face"]["errors"] == 1
    assert snap["requests"]["embed_face"]["total"] == 1


def test_record_face_counts():
    sc = StatsCollector()
    sc.record("embed_face", latency_ms=100.0, num_faces=3, error=False)
    sc.record("embed_face", latency_ms=100.0, num_faces=0, error=False)
    snap = sc.get_snapshot()
    assert snap["faces"]["detected"] == 3
    assert snap["faces"]["no_face_images"] == 1


def test_latency_percentiles():
    sc = StatsCollector()
    for ms in range(1, 101):  # 100 samples: 1ms..100ms
        sc.record("embed_face", latency_ms=float(ms), num_faces=1, error=False)
    lat = sc.get_snapshot()["latency_ms"]["embed_face"]
    assert lat["count"] == 100
    assert 40 <= lat["p50"] <= 60
    assert 85 <= lat["p95"] <= 100


def test_multiple_endpoints_tracked_independently():
    sc = StatsCollector()
    sc.record("embed_face", latency_ms=200.0, num_faces=1, error=False)
    sc.record("detect_face", latency_ms=50.0, num_faces=2, error=False)
    snap = sc.get_snapshot()
    assert snap["requests"]["embed_face"]["total"] == 1
    assert snap["requests"]["detect_face"]["total"] == 1


def test_system_info_present():
    sc = StatsCollector()
    snap = sc.get_snapshot()
    assert "rss_mb" in snap["system"]
    assert "pid" in snap["system"]
    assert "python_version" in snap["system"]
    assert snap["system"]["pid"] > 0
