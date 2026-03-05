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

    def __init__(self, num_jitters: int = 1, model: str = "small") -> None:
        import face_recognition

        self._num_jitters = num_jitters
        self._model = model
        self._face_recognition = face_recognition

    def embed_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """Encode a single face crop. Expects RGB."""
        encodings = self._face_recognition.face_encodings(
            rgb_face,
            known_face_locations=[(0, rgb_face.shape[1], rgb_face.shape[0], 0)],
            num_jitters=self._num_jitters,
            model=self._model,
        )
        if not encodings:
            raise ValueError("No face encoding produced for crop")
        out = np.array(encodings[0], dtype=np.float64)
        assert out.ndim == 1 and out.shape[0] == 128
        return out

    @property
    def embedding_dim(self) -> int:
        return 128


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two embedding vectors (L2)."""
    return float(np.linalg.norm(a - b))
