"""ArcFace ONNX embedding backend (InsightFace w600k_r50).

- Replaces dlib FaceRecognitionEmbedder entirely.
- Implements the same EmbeddingBackend protocol (embed_face interface).
- Produces 512-d L2-normalized embeddings.
- Uses cosine similarity for matching (dot product of normalized vectors).
- Performs 5-point similarity alignment before the 112x112 resize.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# InsightFace w600k_r50 — ArcFace R50 trained on WebFace600K.
# Best balance of accuracy and speed. buffalo_l uses this same backbone.
ARCFACE_MODEL_URL = (
    "https://huggingface.co/deepinsight/insightface/resolve/main/"
    "models/buffalo_l/w600k_r50.onnx"
)

# ArcFace canonical 5-point coordinates for 112x112 aligned output.
# Order: left_eye, right_eye, nose, mouth_left, mouth_right.
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _get_model_path() -> Path:
    """Return path to ArcFace ONNX; download if missing."""
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / "w600k_r50.onnx"
    if not path.exists():
        print(f"Downloading ArcFace model to {path} ...")
        urllib.request.urlretrieve(ARCFACE_MODEL_URL, path)
        print("ArcFace model downloaded.")
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
    """Cosine similarity between two L2-normalized embeddings.

    Since both vectors are already L2-normalized, this is the dot product.
    Range: [-1, 1]. Higher = more similar.
    Matching threshold: similarity >= FACE_MATCH_THRESHOLD (e.g. 0.40).
    """
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
