"""FastAPI entry point for face detection and matching service."""
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import os
from app import detector_mediapipe, io_image
from app import embedder
import app.stats as _stats_mod
from app.logging_config import setup_logging
from app.middleware import RequestLoggingMiddleware, APIKeyMiddleware

_startup_time: float = 0.0
FACE_SERVICE_API_KEY = os.getenv("FACE_SERVICE_API_KEY")

_embedder: "embedder.FaceRecognitionEmbedder | None" = None


def _get_embedder() -> "embedder.FaceRecognitionEmbedder":
    global _embedder
    if _embedder is None:
        _embedder = embedder.FaceRecognitionEmbedder()
    return _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    setup_logging()
    log = structlog.get_logger()
    _startup_time = time.time()

    # Pre-warm singletons so the first real request pays no init cost.
    # These calls run after gunicorn fork — safe for TFLite and dlib.
    detector_mediapipe._get_detector()   # loads MediaPipe TFLite model
    _get_embedder()                      # triggers face_recognition lazy model load
    log.info("face-service started — models pre-warmed", version="0.1.0")
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
        "version": "0.1.0",
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
    """Detect faces and return raw 128-d embeddings.
    Required by SEG workers for background matching.
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

        # SOTA Subject Filtering: Only keep faces > 15% of largest 
        # AND at least 0.5% of the total image area (prevents background crowd indexing)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        largest_area = max(b.area for b in face_boxes)
        
        filtered_boxes = []
        for b in face_boxes:
            is_prominent = b.area >= (largest_area * 0.15)
            is_not_tiny = b.area >= (total_pixels * 0.005) # 0.5% of image (approx 100x100 on 2MP)
            if is_prominent and is_not_tiny:
                filtered_boxes.append(b)
        
        face_boxes = sorted(filtered_boxes, key=lambda b: b.area, reverse=True)

        embedder_instance = _get_embedder()
        results = []

        for box in face_boxes:
            # Match the worker's expected FaceResult interface:
            # { embedding: number[], box: { x, y, w, h } }
            try:
                face_crop = io_image.crop_face_region(img_rgb, box)
                # TTA: Average of original and mirrored
                emb1 = embedder_instance.embed_face(face_crop)
                emb2 = embedder_instance.embed_face(face_crop[:, ::-1])
                embedding = (emb1 + emb2) / 2.0
                
                results.append({
                    "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                    "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
                })
            except Exception as e:
                log.warning("facet-embedding-failed", error=str(e))
                continue

        num_faces = len(results)
        log.info("embed-face", num_faces=num_faces)
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
    tolerance: float = 0.55,
    min_area_ratio: float = 0.15,
):
    """Match faces with SOTA Dynamic Tolerance and TTA.
    
    Args:
        reference: Reference image
        target: Target image
        tolerance: Base tolerance (default 0.52)
        min_area_ratio: Min ratio to largest face (default 0.15)
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
        ref_crop = io_image.crop_face_region(ref_rgb, ref_box)

        embedder_instance = _get_embedder()
        # Reference TTA
        emb_ref1 = embedder_instance.embed_face(ref_crop)
        emb_ref2 = embedder_instance.embed_face(ref_crop[:, ::-1])
        ref_embedding = (emb_ref1 + emb_ref2) / 2.0

        target_bytes = await target.read()
        target_bgr = io_image.load_image_from_bytes(target_bytes)
        if target_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load target image")
        target_rgb = io_image.bgr_to_rgb(target_bgr)

        target_boxes = detector_mediapipe.detect_faces(target_rgb)
        num_faces = len(target_boxes)
        if not target_boxes:
             return JSONResponse(content={"matched": False, "best_distance": 1.0, "num_faces": 0})

        largest_area = max(box.area for box in target_boxes)
        best_distance = float("inf")
        matched = False

        for box in target_boxes:
            # Area Filtering (SOTA Crowd-Proofing)
            if box.area < (largest_area * min_area_ratio):
                continue
                
            target_crop = io_image.crop_face_region(target_rgb, box)
            try:
                # Target TTA
                emb_t1 = embedder_instance.embed_face(target_crop)
                emb_t2 = embedder_instance.embed_face(target_crop[:, ::-1])
                target_embedding = (emb_t1 + emb_t2) / 2.0
                
                distance = embedder.euclidean_distance(ref_embedding, target_embedding)
                area_ratio = box.area / largest_area
                effective_tolerance = tolerance - (1.0 - area_ratio) * 0.25
                
                if distance < best_distance:
                    best_distance = distance
                if distance <= effective_tolerance:
                    matched = True
            except Exception:
                continue

        log.info("match-face", matched=matched, best_distance=best_distance, tolerance=tolerance)
        return JSONResponse(content={
            "matched": matched,
            "best_distance": best_distance,
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
