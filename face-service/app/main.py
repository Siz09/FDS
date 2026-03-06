"""FastAPI entry point for face detection and matching service."""
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app import detector_mediapipe, io_image
from app import embedder
from app.logging_config import setup_logging
from app.middleware import RequestLoggingMiddleware

_startup_time: float = 0.0
_embedder = embedder.FaceRecognitionEmbedder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    setup_logging()
    log = structlog.get_logger()
    _startup_time = time.time()
    log.info("face-service started", version="0.1.0")
    yield
    log.info("face-service shutting down")


app = FastAPI(title="Face Service", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "face-service",
        "version": "0.1.0",
        "uptime": round(time.time() - _startup_time, 2),
    }


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
    try:
        image_bytes = await image.read()
        img_bgr = io_image.load_image_from_bytes(image_bytes)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        img_rgb = io_image.bgr_to_rgb(img_bgr)
        face_boxes = detector_mediapipe.detect_faces(
            img_rgb,
            min_detection_confidence=min_detection_confidence,
            max_faces=max_faces,
        )

        result = [
            {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
            for box in face_boxes
        ]

        log.info("detect-face", num_faces=len(face_boxes))
        return JSONResponse(content={"num_faces": len(face_boxes), "faces": result})

    except HTTPException:
        raise
    except Exception as e:
        log.error("detect-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/embed-face")
async def embed_face(image: UploadFile = File(...)):
    """Generate 128-d face embeddings for all faces detected in an uploaded image.

    Returns:
        JSON with list of faces, each containing a 128-d embedding and bounding box
    """
    log = structlog.get_logger()
    try:
        image_bytes = await image.read()
        img_bgr = io_image.load_image_from_bytes(image_bytes)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        img_rgb = io_image.bgr_to_rgb(img_bgr)
        face_boxes = detector_mediapipe.detect_faces(img_rgb)

        faces = []
        for box in face_boxes:
            crop = io_image.crop_face_region(img_rgb, box)
            try:
                emb = _embedder.embed_face(crop)
                faces.append({
                    "embedding": emb.tolist(),
                    "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                })
            except ValueError:
                continue

        log.info("embed-face", num_faces=len(faces))
        return JSONResponse(content={"faces": faces})

    except HTTPException:
        raise
    except Exception as e:
        log.error("embed-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/match-face")
async def match_face(
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    tolerance: float = 0.6,
):
    """Match faces between two images.

    Args:
        reference: Reference image with the face to match
        target: Target image to search for the face
        tolerance: Maximum Euclidean distance to consider a match (lower = stricter)

    Returns:
        JSON with match result and distance
    """
    log = structlog.get_logger()
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

        embedder_instance = embedder.FaceRecognitionEmbedder()
        ref_embedding = embedder_instance.embed_face(ref_crop)

        target_bytes = await target.read()
        target_bgr = io_image.load_image_from_bytes(target_bytes)
        if target_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to load target image")
        target_rgb = io_image.bgr_to_rgb(target_bgr)

        target_boxes = detector_mediapipe.detect_faces(target_rgb)

        best_distance = float("inf")
        matched = False

        for box in target_boxes:
            target_crop = io_image.crop_face_region(target_rgb, box)
            try:
                target_embedding = embedder_instance.embed_face(target_crop)
                distance = embedder.euclidean_distance(ref_embedding, target_embedding)
                if distance < best_distance:
                    best_distance = distance
                if distance <= tolerance:
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
        log.error("match-face failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error matching faces: {str(e)}")
