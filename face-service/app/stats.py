"""In-process metrics collector for face-service.

Thread-safe singleton. Stats are per-process — under Gunicorn with multiple workers,
each worker has its own instance. The /stats response includes pid so callers can
identify which worker responded. Do not treat counts as service totals.

Latency percentiles use reservoir sampling (1000-slot window) to bound memory use.
"""
import os
import random
import sys
import threading
import time
from typing import Optional

import psutil

_RESERVOIR_SIZE = 1000


class StatsCollector:
    """Per-endpoint request counts, face counts, and latency percentiles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._process = psutil.Process(os.getpid())
        self._requests: dict[str, dict[str, int]] = {}
        self._faces_detected: int = 0
        self._no_face_images: int = 0
        self._reservoirs: dict[str, dict] = {}

    def record(self, endpoint: str, latency_ms: float, num_faces: int, error: bool) -> None:
        """Record one request outcome. Call from endpoint finally blocks."""
        with self._lock:
            if endpoint not in self._requests:
                self._requests[endpoint] = {"total": 0, "errors": 0}
            self._requests[endpoint]["total"] += 1
            if error:
                self._requests[endpoint]["errors"] += 1

            self._faces_detected += num_faces
            if num_faces == 0 and not error:
                self._no_face_images += 1

            if endpoint not in self._reservoirs:
                self._reservoirs[endpoint] = {"samples": [], "count": 0}
            r = self._reservoirs[endpoint]
            r["count"] += 1
            n = r["count"]
            if len(r["samples"]) < _RESERVOIR_SIZE:
                r["samples"].append(latency_ms)
            else:
                j = random.randint(0, n - 1)
                if j < _RESERVOIR_SIZE:
                    r["samples"][j] = latency_ms

    def get_snapshot(self) -> dict:
        """Point-in-time snapshot. Safe to call from any thread."""
        with self._lock:
            uptime = round(time.time() - self._start_time, 1)
            requests_copy = {ep: dict(c) for ep, c in self._requests.items()}
            faces_copy = {"detected": self._faces_detected, "no_face_images": self._no_face_images}
            latency_copy: dict[str, dict] = {}
            for ep, r in self._reservoirs.items():
                samples = sorted(r["samples"])
                latency_copy[ep] = {
                    "p50": round(_pct(samples, 50), 1) if samples else 0.0,
                    "p95": round(_pct(samples, 95), 1) if samples else 0.0,
                    "count": r["count"],
                }

        try:
            rss_mb: Optional[float] = round(self._process.memory_info().rss / (1024 * 1024), 1)
        except psutil.NoSuchProcess:
            rss_mb = None

        return {
            "uptime_s": uptime,
            "requests": requests_copy,
            "faces": faces_copy,
            "latency_ms": latency_copy,
            "system": {"rss_mb": rss_mb, "pid": os.getpid(), "python_version": sys.version.split()[0]},
        }


def _pct(sorted_samples: list[float], p: int) -> float:
    n = len(sorted_samples)
    return sorted_samples[max(0, int((p / 100.0) * n) - 1)]


# Import as: import app.stats as _stats_mod  →  _stats_mod.stats.record(...)
# Not: from app.stats import stats  (breaks test fixture resets)
stats = StatsCollector()
