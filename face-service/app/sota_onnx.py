"""ArcFace ONNX embedding backend (InsightFace w600k_r50).

- Replaces dlib FaceRecognitionEmbedder entirely.
- Implements the same EmbeddingBackend protocol (embed_face interface).
- Produces 512-d L2-normalized embeddings.
- Uses cosine similarity for matching (dot product of normalized vectors).
- Performs 5-point similarity alignment before the 112x112 resize.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import structlog

# ArcFace canonical 5-point coordinates for 112x112 aligned output.
# Order: left_eye, right_eye, nose, mouth_left, mouth_right.
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# These are the indices within the 106-point output that correspond to the
# 5 canonical ArcFace landmarks: left_eye, right_eye, nose_tip, mouth_left, mouth_right.
# Source: InsightFace 2d106det landmark ordering.
_LM106_5PT_IDX = [38, 88, 86, 52, 61]

# Expected model path (baked into Docker image at build time — no runtime download).
# See Dockerfile: the RUN step installs the ONNX file to /app/models/w600k_r50.onnx
# before COPY . . so it is embedded in the image layer.
_MODEL_FILENAME = "w600k_r50.onnx"
_LANDMARK_MODEL_FILENAME = "2d106det.onnx"


def _get_model_path() -> Path:
    """Return path to ArcFace ONNX model.

    The model is baked into the Docker image at build time (see Dockerfile).
    This function does NOT download anything at runtime — failing loudly here
    is better than a silent 401 / corrupted file on a remote host with no HF token.
    """
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "models" / _MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"ArcFace model not found at {path}. "
            "The model must be baked into the Docker image. "
            "Rebuild the image with: docker compose build --no-cache face-service"
        )
    return path


def _get_landmark_model_path() -> Path:
    """Return path to 2d106det landmark ONNX model.

    The model is baked into the Docker image at build time (see Dockerfile).
    This function does NOT download anything at runtime.
    """
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "models" / _LANDMARK_MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"2d106det landmark model not found at {path}. "
            "Place 2d106det.onnx in models/ and rebuild the Docker image."
        )
    return path


def align_face_5pt(image: np.ndarray, landmarks: list[tuple]) -> np.ndarray | None:
    """Warp face to ArcFace canonical 112x112 using 5-point similarity transform.

    Args:
        image: Full RGB image (H, W, 3) uint8.
        landmarks: 5 points [eye_left, eye_right, nose, mouth_left, mouth_right]
                   as (x, y) pixel tuples, in that exact order.

    Returns:
        112x112 RGB aligned face crop, or None if the transform fails.
    """
    src = np.array(landmarks, dtype=np.float32)
    # estimateAffinePartial2D: similarity transform (rotation + uniform scale + translation).
    # No shear — preserves face geometry. Standard for ArcFace alignment.
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(image, M, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned


def align_face_106pt(image: np.ndarray, landmarks_106: np.ndarray) -> np.ndarray | None:
    """Warp face to ArcFace canonical 112x112 using 106-point derived landmarks.

    Extracts the 5 most geometrically stable points from the 106-point output
    and uses the same affine similarity transform as align_face_5pt.
    More accurate than align_face_5pt on non-frontal faces (yaw/pitch).

    Args:
        image: Full RGB image (H, W, 3).
        landmarks_106: (106, 2) float32 array of landmark coordinates
                       in original image pixel space.

    Returns:
        112x112 RGB aligned face crop, or None if transform estimation fails.
    """
    src = landmarks_106[_LM106_5PT_IDX].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(image, M, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned


_arcface_embedder: "ArcFaceEmbedder | None" = None


def get_arcface_embedder() -> "ArcFaceEmbedder":
    """Process-level singleton — mirrors _get_embedder() pattern in main.py."""
    global _arcface_embedder
    if _arcface_embedder is None:
        _arcface_embedder = ArcFaceEmbedder()
    return _arcface_embedder


class ArcFaceEmbedder:
    """ArcFace R50 ONNX embedding backend.

    Implements EmbeddingBackend protocol: embed_face(rgb_face) -> (512,) float32.
    Input is a pre-cropped/aligned face (RGB). Internally resizes to 112x112.
    Output is L2-normalized — use cosine similarity (dot product) for matching.
    """

    def __init__(self) -> None:
        model_path = _get_model_path()
        # CPUExecutionProvider only — GPU not required for production load.
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def embed_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """Compute 512-d L2-normalized embedding for a single aligned face crop.

        Args:
            rgb_face: Face crop in RGB, any size. Should already be aligned
                      (112x112 from align_face_5pt) but handles other sizes.

        Returns:
            (512,) float32 array, L2-normalized.
        """
        img = cv2.resize(rgb_face, (112, 112))
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0          # normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # add batch dim -> (1, 3, 112, 112)

        output = self.session.run(None, {self.input_name: img})[0]  # (1, 512)
        embedding = output[0]                 # (512,)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        return (embedding / (norm + 1e-6)).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return 512


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two L2-normalized embeddings.

    Both inputs MUST already be L2-normalized (embed_face guarantees this).
    For unit vectors, cosine similarity == dot product — no norm recompute needed.
    Range: [-1, 1]. Higher = more similar.
    Matching threshold: similarity >= FACE_MATCH_THRESHOLD (e.g. 0.40).
    """
    return float(np.dot(a, b))


class LandmarkDetector106:
    """InsightFace 2d106det — 106-point facial landmark detector.

    Takes a loose bounding box crop from the full image and returns
    106 landmark coordinates mapped back to original image pixel space.
    Input size: 192x192. Output: (106, 2) array of (x, y) pixel coords.
    """

    INPUT_SIZE = (192, 192)

    def __init__(self) -> None:
        model_path = _get_landmark_model_path()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def get_landmarks(self, image: np.ndarray, bbox) -> np.ndarray | None:
        """Detect 106 landmarks for one face.

        Args:
            image: Full RGB image (H, W, 3), uint8.
            bbox: FaceBox with .x .y .w .h in pixel coords.

        Returns:
            (106, 2) float32 array of (x, y) coords in original image space,
            or None if inference fails.
        """
        img_h, img_w = image.shape[:2]

        # Expand bbox by 40% on each side to give the landmark model full head structure context.
        pad_x = int(bbox.w * 0.4)
        pad_y = int(bbox.h * 0.4)
        x1 = max(0, bbox.x - pad_x)
        y1 = max(0, bbox.y - pad_y)
        x2 = min(img_w, bbox.x + bbox.w + pad_x)
        y2 = min(img_h, bbox.y + bbox.h + pad_y)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_h, crop_w = crop.shape[:2]

        # Resize to 192x192 and normalize to [-1, 1]
        resized = cv2.resize(crop, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 127.5 - 1.0
        blob = np.transpose(blob, (2, 0, 1))        # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)          # add batch dim

        try:
            output = self.session.run(None, {self.input_name: blob})[0]
            structlog.get_logger().info("2d106det-output-shape", shape=str(output.shape))
        except Exception as e:
            structlog.get_logger().error("2d106det inference failed", error=str(e))
            return None

        # output shape is (1, 212) — 106 points * 2 coords, flattened.
        # Values are normalized in [-1, 1] relative to the crop!
        landmarks = output[0].reshape(106, 2)

        # Step 1: convert normalized [-1, 1] to normalized [0, 1] inside crop
        landmarks = (landmarks + 1.0) / 2.0
        # Step 2: scale by crop dimensions
        landmarks[:, 0] = landmarks[:, 0] * crop_w
        landmarks[:, 1] = landmarks[:, 1] * crop_h
        # Step 3: offset from crop origin to full image coords
        landmarks[:, 0] += x1
        landmarks[:, 1] += y1

        return landmarks.astype(np.float32)


_landmark_detector: LandmarkDetector106 | None = None


def get_landmark_detector() -> LandmarkDetector106:
    """Process-level singleton for the 106-point landmark detector."""
    global _landmark_detector
    if _landmark_detector is None:
        _landmark_detector = LandmarkDetector106()
    return _landmark_detector
