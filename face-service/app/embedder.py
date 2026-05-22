"""Face embedding backend protocol.

The concrete implementation is ArcFaceEmbedder in sota_onnx.py.
This module keeps only the shared Protocol so other modules can type-check
against the interface without importing ArcFace directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface for face embedding backends.

    ArcFaceEmbedder in sota_onnx.py satisfies this protocol.
    Produces L2-normalized embeddings — use cosine similarity for matching.
    """

    def embed_face(self, rgb_face: np.ndarray) -> np.ndarray:
        """Compute embedding for a single face crop (RGB).

        Args:
            rgb_face: Face region in RGB, any size (backend resizes to 112x112).

        Returns:
            1-d float32 array, shape (D,), L2-normalized.
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding vector (512 for ArcFace R50)."""
        ...
