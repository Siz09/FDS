"""Face embedding backend: 128-d identity vector per face.

- EmbeddingBackend interface: embed_face(rgb_face) -> (D,) float array.
- Default: face_recognition (dlib) 128-d encoding.
- Euclidean distance used for matching (face_recognition convention).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

if __name__ != "__main__":
    pass
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface for face embedding backends."""

    def embed_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """Compute embedding for a single face crop (RGB).

        Args:
            rgb_face: Face region in RGB, any size (backend may resize).

        Returns:
            1-d array of dtype float32 or float64, shape (D,).
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding vector (e.g. 128)."""
        ...


class FaceRecognitionEmbedder:
    """face_recognition (dlib) 128-d encoding backend."""

    def __init__(self, num_jitters: int = 1, model: str = "large") -> None:
        import face_recognition

        self._num_jitters = num_jitters
        self._model = model
        self._face_recognition = face_recognition

    def embed_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """Encode a single face crop. Expects RGB. Uses Multi-Crop if jitters are high."""
        import cv2

        def get_encoding(img_crop, jitters):
            # Dlib requires C-contiguous uint8 arrays
            if not img_crop.flags.c_contiguous:
                img_crop = np.ascontiguousarray(img_crop)
            if img_crop.dtype != np.uint8:
                img_crop = img_crop.astype(np.uint8)

            encs = self._face_recognition.face_encodings(
                img_crop,
                known_face_locations=[(0, img_crop.shape[1], img_crop.shape[0], 0)],
                num_jitters=jitters,
                model=self._model,
            )
            return encs[0] if encs else None

        # Simple 1-crop if low jitters (up to 25)
        if self._num_jitters <= 25:
            e = get_encoding(rgb_face, self._num_jitters)
            if e is None: raise ValueError("No face encoding")
            return np.array(e, dtype=np.float64)

        # Multi-Crop Strategy (TTA): Original + Mirrored
        # To avoid explosive complexity, we distribute jitters across crops.
        # If user asked for 100 jitters, we do 2 crops x 50 jitters each.
        jitters_per_crop = max(1, self._num_jitters // 2)
        crops = [rgb_face, cv2.flip(rgb_face, 1)]
        
        all_encs = []
        for c in crops:
            e = get_encoding(c, jitters_per_crop)
            if e is not None:
                all_encs.append(e)

        if not all_encs:
            raise ValueError("No face encoding produced for any crop")

        # Average the embeddings for maximum stability
        avg_enc = np.mean(all_encs, axis=0)
        return np.array(avg_enc, dtype=np.float64)

    @property
    def embedding_dim(self) -> int:
        return 128


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two embedding vectors (L2)."""
    return float(np.linalg.norm(a - b))
