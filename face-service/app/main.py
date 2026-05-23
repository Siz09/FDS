"""FastAPI entry point for face detection and matching service."""
import asyncio
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import structlog
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import os
from app import detector_mediapipe, io_image
from app.sota_onnx import (
    get_arcface_embedder,
    get_landmark_detector,
    align_face_5pt,
    align_face_106pt,
    cosine_similarity,
)
import app.stats as _stats_mod
from app.logging_config import setup_logging
from app.middleware import RequestLoggingMiddleware, APIKeyMiddleware

# Thread pool for CPU-bound ArcFace ONNX inference.
# One thread per gunicorn worker — ONNX is CPU-bound, contention hurts throughput.
# Set EMBED_THREAD_WORKERS=2 only on machines with spare CPU cores.
_THREAD_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("EMBED_THREAD_WORKERS", "1")))

_startup_time: float = 0.0
FACE_SERVICE_API_KEY = os.getenv("FACE_SERVICE_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    setup_logging()
    log = structlog.get_logger()
    _startup_time = time.time()

    # Pre-warm singletons so the first real request pays no init cost.
    # These calls run after gunicorn fork — safe for TFLite and ONNX.
    detector_mediapipe._get_detector()   # Load MediaPipe TFLite
    get_arcface_embedder()               # Load ArcFace ONNX (w600k_r50)
    get_landmark_detector()              # Load 106-point landmark ONNX (2d106det)
    log.info("face-service started — models pre-warmed", version="0.2.0")
    yield
    log.info("face-service shutting down")


app = FastAPI(title="Face Service", lifespan=lifespan)
app.add_middleware(APIKeyMiddleware, api_key=FACE_SERVICE_API_KEY)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "face-service",
        "version": "0.2.0",
        "uptime_s": round(time.time() - _startup_time, 2),
    }


@app.get("/stats")
async def get_stats():
    """Per-process performance stats. Under Gunicorn, each worker has its own instance.
    Check system.pid to identify which worker responded."""
    return _stats_mod.stats.get_snapshot()


@app.post("/detect-face")
async def detect_face(
    image: UploadFile = File(...),
    min_detection_confidence: float = 0.5,
    max_faces: int = 10,
):
    """Detect faces in an uploaded image.

    Args:
        image: Image file (JPEG, PNG, etc.)
        min_detection_confidence: Minimum confidence threshold [0, 1]
        max_faces: Maximum number of faces to return

    Returns:
        JSON with list of detected face bounding boxes
    """
    log = structlog.get_logger()
    t0 = time.time()
    error = False
    num_faces = 0
    try:
        image_bytes = await image.read()
        img_bgr = io_image.load_image_from_bytes(image_bytes)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        img_rgb = io_image.bgr_to_rgb(img_bgr)
        face_boxes = detector_mediapipe.detect_faces(
            img_rgb,
            max_faces=max_faces,
        )

        result = [
            {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
            for box in face_boxes
        ]

        num_faces = len(face_boxes)
        log.info("detect-face", num_faces=num_faces)
        return JSONResponse(content={"num_faces": num_faces, "faces": result})

    except HTTPException:
        raise
    except Exception as e:
        error = True
        log.error("detect-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        _stats_mod.stats.record("detect_face", latency_ms=(time.time() - t0) * 1000, num_faces=num_faces, error=error)


@app.post("/embed-face")
async def embed_face(
    image: UploadFile = File(...),
    max_faces: int = 20,
):
    """Detect faces and return raw 512-d ArcFace embeddings.
    Required by SEG workers for background matching.
    Uses tensor batching — all faces are aligned concurrently, then embedded
    in a single ONNX inference call for 3-5x throughput on multi-face photos.
    """
    log = structlog.get_logger()
    t0 = time.time()
    error = False
    num_faces = 0
    try:
        image_bytes = await image.read()
        img_bgr = io_image.load_image_from_bytes(image_bytes)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        img_rgb = io_image.bgr_to_rgb(img_bgr)
        face_boxes = detector_mediapipe.detect_faces(
            img_rgb,
            max_faces=max_faces,
        )

        if not face_boxes:
             return JSONResponse(content={"faces": []})

        # Absolute minimum: face must be at least 40x40 px to yield a usable embedding.
        filtered_boxes = [b for b in face_boxes if b.w >= 40 and b.h >= 40]
        face_boxes = sorted(filtered_boxes, key=lambda b: b.area, reverse=True)

        if not face_boxes:
            return JSONResponse(content={"faces": []})

        embedder_instance = get_arcface_embedder()
        landmark_detector = get_landmark_detector()
        loop = asyncio.get_running_loop()

        async def _align_box(box):
            """Phase 1: align face crop only — no embedding yet."""
            lm106 = await loop.run_in_executor(
                _THREAD_POOL, landmark_detector.get_landmarks, img_rgb, box
            )
            if lm106 is not None:
                face_crop = align_face_106pt(img_rgb, lm106)
                if face_crop is not None:
                    log.info("align-path", path="106pt")
                else:
                    face_crop = align_face_5pt(img_rgb, box.landmarks) if box.landmarks else io_image.crop_face_region(img_rgb, box)
                    log.info("align-path", path="5pt-fallback-from-106pt")
            elif box.landmarks:
                face_crop = align_face_5pt(img_rgb, box.landmarks)
                if face_crop is None:
                    face_crop = io_image.crop_face_region(img_rgb, box)
                    log.info("align-path", path="bbox-fallback")
                else:
                    log.info("align-path", path="5pt")
            else:
                face_crop = io_image.crop_face_region(img_rgb, box)
                log.info("align-path", path="bbox-only")
            return face_crop, box

        # Phase 1: align all faces concurrently (landmark detection is I/O-like on CPU).
        align_tasks = await asyncio.gather(
            *[_align_box(box) for box in face_boxes],
            return_exceptions=True,
        )

        # Collect successfully aligned crops, preserving box association.
        aligned_crops = []
        aligned_boxes = []
        for outcome in align_tasks:
            if isinstance(outcome, Exception):
                log.warning("face-alignment-failed", error=str(outcome))
                continue
            crop, box = outcome
            if crop is not None:
                aligned_crops.append(crop)
                aligned_boxes.append(box)

        if not aligned_crops:
            return JSONResponse(content={"faces": []})

        # Phase 2: single batched ONNX call for all aligned crops — 3-5x faster than N calls.
        embeddings = await loop.run_in_executor(
            _THREAD_POOL, embedder_instance.embed_batch, aligned_crops
        )

        results = []
        for (embedding, quality_score), box in zip(embeddings, aligned_boxes):
            results.append({
                "embedding": embedding.tolist(),
                "quality_score": quality_score,
                "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
            })

        num_faces = len(results)
        log.info("embed-face", num_faces=num_faces, batched=True)
        return JSONResponse(content={"faces": results})

    except HTTPException:
        raise
    except Exception as e:
        error = True
        log.error("embed-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error embedding faces: {str(e)}")
    finally:
        _stats_mod.stats.record("embed_face", latency_ms=(time.time() - t0) * 1000, num_faces=num_faces, error=error)


@app.post("/match-face")
async def match_face(
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    tolerance: float = 0.40,
):
    """Match faces using ArcFace cosine similarity.

    Args:
        reference: Reference image (selfie).
        target: Target image (event photo).
        tolerance: Cosine similarity threshold; >= means match (default 0.40).
    """
    log = structlog.get_logger()
    t0 = time.time()
    error = False
    num_faces = 0
    try:
        ref_bytes = await reference.read()
        ref_bgr = io_image.load_image_from_bytes(ref_bytes)
        if ref_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load reference image")
        ref_rgb = io_image.bgr_to_rgb(ref_bgr)

        ref_boxes = detector_mediapipe.detect_faces(ref_rgb)
        if not ref_boxes:
            raise HTTPException(status_code=400, detail="No face detected in reference image")

        ref_box = detector_mediapipe.get_largest_face(ref_boxes)

        landmark_detector = get_landmark_detector()
        ref_lm106 = landmark_detector.get_landmarks(ref_rgb, ref_box)
        if ref_lm106 is not None:
            ref_crop = align_face_106pt(ref_rgb, ref_lm106)
            if ref_crop is None:
                ref_crop = align_face_5pt(ref_rgb, ref_box.landmarks) if ref_box.landmarks else io_image.crop_face_region(ref_rgb, ref_box)
        elif ref_box.landmarks:
            ref_crop = align_face_5pt(ref_rgb, ref_box.landmarks)
            if ref_crop is None:
                ref_crop = io_image.crop_face_region(ref_rgb, ref_box)
        else:
            ref_crop = io_image.crop_face_region(ref_rgb, ref_box)

        embedder_instance = get_arcface_embedder()
        ref_embedding, _ = embedder_instance.embed_face(ref_crop)  # quality_score not needed for ref

        target_bytes = await target.read()
        target_bgr = io_image.load_image_from_bytes(target_bytes)
        if target_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load target image")
        target_rgb = io_image.bgr_to_rgb(target_bgr)

        target_boxes = detector_mediapipe.detect_faces(target_rgb)
        num_faces = len(target_boxes)
        if not target_boxes:
             return JSONResponse(content={
                 "matched": False,
                 "best_similarity": -1.0,
                 "num_faces": 0,
             })

        # ArcFace with alignment is robust to face size — no dynamic tolerance needed.
        best_similarity = -1.0
        matched = False

        for box in target_boxes:
            # Absolute size filter (40x40px minimum)
            if box.w < 40 or box.h < 40:
                continue

            # 106-point landmark alignment (landmark_detector already obtained above).
            target_lm106 = landmark_detector.get_landmarks(target_rgb, box)
            if target_lm106 is not None:
                target_crop = align_face_106pt(target_rgb, target_lm106)
                if target_crop is None:
                    target_crop = align_face_5pt(target_rgb, box.landmarks) if box.landmarks else io_image.crop_face_region(target_rgb, box)
            elif box.landmarks:
                target_crop = align_face_5pt(target_rgb, box.landmarks)
                if target_crop is None:
                    target_crop = io_image.crop_face_region(target_rgb, box)
            else:
                target_crop = io_image.crop_face_region(target_rgb, box)

            try:
                target_embedding, _ = embedder_instance.embed_face(target_crop)  # quality_score unused in match-face
                similarity = cosine_similarity(ref_embedding, target_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                # ArcFace cosine: higher = more similar. >= threshold means match.
                if similarity >= tolerance:
                    matched = True
            except Exception:
                continue

        log.info("match-face", matched=matched, best_similarity=best_similarity, tolerance=tolerance)
        return JSONResponse(content={
            "matched": matched,
            "best_similarity": best_similarity,
            "tolerance": tolerance,
            "num_faces_in_target": len(target_boxes),
        })

    except HTTPException:
        raise
    except Exception as e:
        error = True
        log.error("match-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error matching faces: {str(e)}")
    finally:
        _stats_mod.stats.record("match_face", latency_ms=(time.time() - t0) * 1000, num_faces=num_faces, error=error)
