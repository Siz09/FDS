"""Benchmark: face detection + embedding latency.

Measures:
  1. Single-image latency (load → detect → embed all faces).
  2. 10-image sequential loop total and per-image average.
  3. Timeout edge-case: URL fetch that never responds (simulated).

Run from python/face-service/ with the venv active:
    python scripts/benchmark.py [--image PATH_OR_URL]

If no image is provided, a synthetic RGB test image is used so the
benchmark runs without any external data dependency.
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Allow running as a script from python/face-service/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import detector_mediapipe, embedder, io_image

BATCH_SIZE = 10
EMBED_TIMEOUT_SECONDS = 30  # wall-clock guard per image
URL_FETCH_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_face_image(size: int = 256) -> np.ndarray:
    """Return a plain RGB uint8 array (no real face — detection returns 0 boxes)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _load_image(source: str | None) -> np.ndarray:
    """Load image from path, URL, or generate synthetic fallback."""
    if source is None:
        print("  [bench] No image provided — using synthetic 256×256 image.")
        return _make_synthetic_face_image()

    if source.startswith(("http://", "https://")):
        print(f"  [bench] Fetching URL (timeout={URL_FETCH_TIMEOUT_SECONDS}s): {source}")
        img = io_image.load_image_from_url(source, timeout=URL_FETCH_TIMEOUT_SECONDS)
    else:
        img = io_image.load_image(source)

    if img is None:
        print("  [bench] WARNING: failed to load image — falling back to synthetic.")
        return _make_synthetic_face_image()
    return img


def _process_image(img_bgr: np.ndarray) -> dict:
    """Detect faces and embed each one. Returns timing + result summary."""
    t0 = time.perf_counter()

    img_rgb = io_image.bgr_to_rgb(img_bgr)
    boxes = detector_mediapipe.detect_faces(img_rgb)

    t_detect = time.perf_counter()

    emb_instance = embedder.FaceRecognitionEmbedder()
    embeddings = []
    for box in boxes:
        crop = io_image.crop_face_region(img_rgb, box)
        try:
            emb = emb_instance.embed_face(crop)
            embeddings.append(emb)
        except ValueError:
            pass  # crop produced no encoding

    t_embed = time.perf_counter()

    return {
        "num_faces": len(boxes),
        "num_embeddings": len(embeddings),
        "detect_ms": (t_detect - t0) * 1000,
        "embed_ms": (t_embed - t_detect) * 1000,
        "total_ms": (t_embed - t0) * 1000,
    }


# ---------------------------------------------------------------------------
# Timeout helper (wall-clock guard via thread)
# ---------------------------------------------------------------------------

def _process_with_timeout(img_bgr: np.ndarray, timeout: float) -> dict | None:
    """Run _process_image in a thread; return None if it exceeds timeout."""
    result: dict | None = None
    exc: BaseException | None = None

    def _run() -> None:
        nonlocal result, exc
        try:
            result = _process_image(img_bgr)
        except Exception as e:
            exc = e

    t = threading.Thread(target=_run, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout)
    elapsed = time.perf_counter() - t0

    if t.is_alive():
        print(f"  [bench] TIMEOUT: processing exceeded {timeout}s wall-clock limit.")
        return None
    if exc is not None:
        raise exc
    return result


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_single(img_bgr: np.ndarray) -> dict:
    print("\n--- Single-image benchmark ---")
    # Warm up the singleton detector (first call loads .tflite model).
    print("  Warming up detector singleton...")
    t_warm0 = time.perf_counter()
    detector_mediapipe._get_detector()
    t_warm1 = time.perf_counter()
    print(f"  Detector init: {(t_warm1 - t_warm0)*1000:.1f} ms (one-time cost)")

    r = _process_with_timeout(img_bgr, timeout=EMBED_TIMEOUT_SECONDS)
    if r is None:
        return {}
    print(f"  Faces found   : {r['num_faces']}")
    print(f"  Detect        : {r['detect_ms']:.1f} ms")
    print(f"  Embed         : {r['embed_ms']:.1f} ms  ({r['num_embeddings']} face(s))")
    print(f"  Total         : {r['total_ms']:.1f} ms")
    return r


def bench_batch(img_bgr: np.ndarray, n: int = BATCH_SIZE) -> None:
    print(f"\n--- {n}-image sequential loop benchmark ---")
    totals: list[float] = []
    for i in range(n):
        r = _process_with_timeout(img_bgr, timeout=EMBED_TIMEOUT_SECONDS)
        if r is None:
            print(f"  Image {i+1}: TIMED OUT")
            continue
        totals.append(r["total_ms"])
        print(f"  Image {i+1:2d}: {r['total_ms']:.1f} ms  ({r['num_faces']} face(s))")

    if totals:
        print(f"  Loop total  : {sum(totals):.1f} ms")
        print(f"  Per-image   : {sum(totals)/len(totals):.1f} ms avg")


def bench_url_timeout() -> None:
    """Show that load_image_from_url respects the timeout param."""
    print("\n--- URL timeout edge-case ---")
    # Use a non-routable address; should time out quickly.
    bad_url = "http://10.255.255.1/fake_image.jpg"
    print(f"  Fetching unreachable URL (timeout={URL_FETCH_TIMEOUT_SECONDS}s): {bad_url}")
    t0 = time.perf_counter()
    img = io_image.load_image_from_url(bad_url, timeout=URL_FETCH_TIMEOUT_SECONDS)
    elapsed = (time.perf_counter() - t0) * 1000
    if img is None:
        print(f"  Result: None (timeout/error) — elapsed {elapsed:.0f} ms  ✓")
    else:
        print(f"  Result: unexpected success — elapsed {elapsed:.0f} ms")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Face service benchmark")
    parser.add_argument(
        "--image",
        default=None,
        help="Path or HTTP URL to a test image (default: synthetic image)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Face Service — Performance Benchmark")
    print("=" * 50)

    img_bgr = _load_image(args.image)
    h, w = img_bgr.shape[:2]
    print(f"  Image size: {w}×{h}")

    bench_single(img_bgr)
    bench_batch(img_bgr)
    bench_url_timeout()

    print("\n" + "=" * 50)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
