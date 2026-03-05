"""Tests for embedding shape and dtype."""

import numpy as np
import pytest

# Add repo root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedder import FaceRecognitionEmbedder, euclidean_distance
from app.types import FaceResult, FaceBox


def test_embedding_dim() -> None:
    pytest.importorskip("face_recognition")
    backend = FaceRecognitionEmbedder(num_jitters=1)
    assert backend.embedding_dim == 128


def test_euclidean_distance_shape() -> None:
    a = np.random.randn(128).astype(np.float64)
    b = np.random.randn(128).astype(np.float64)
    d = euclidean_distance(a, b)
    assert isinstance(d, float)
    assert d >= 0
    assert euclidean_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_face_result_embedding_validation() -> None:
    box = FaceBox(x=0, y=0, w=64, h=64)
    # Valid 1-d float64
    FaceResult(bbox=box, embedding=np.zeros(128, dtype=np.float64))
    # Valid float32
    FaceResult(bbox=box, embedding=np.zeros(128, dtype=np.float32))
    # Invalid: 2-d
    with pytest.raises(ValueError, match="1-d"):
        FaceResult(bbox=box, embedding=np.zeros((128, 1), dtype=np.float64))
    # Invalid dtype
    with pytest.raises(ValueError, match="float"):
        FaceResult(bbox=box, embedding=np.zeros(128, dtype=np.int32))


def test_embed_face_integration_requires_face() -> None:
    """Embedder.embed_face on random RGB may fail (no face); we only check it raises or returns 128-d."""
    pytest.importorskip("face_recognition")
    backend = FaceRecognitionEmbedder(num_jitters=1)
    # Random 64x64 RGB - no face; face_recognition may return [] and we'd raise in embed_face
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    try:
        emb = backend.embed_face(rgb)
        assert emb.shape == (128,)
        assert emb.dtype in (np.float32, np.float64)
    except ValueError:
        # Expected when no face in image
        pass
