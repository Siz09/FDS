#!/usr/bin/env python3
"""Enroll Person-1 from a single reference image and save embedding (NPY).

Usage:
  python scripts/enroll.py --name person1 --image data/person1.jpg --out known/person1.npy
  python scripts/enroll.py --name person1 --image data/person1.jpg --out known/person1.npy --num-jitters 2
"""

import argparse
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.detector_mediapipe import detect_faces, get_largest_face
from app.embedder import FaceRecognitionEmbedder
from app.io_image import bgr_to_rgb, crop_face_region, load_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enroll a person from one reference image; save 128-d embedding as NPY."
    )
    parser.add_argument("--name", required=True, help="Label for the person (e.g. person1)")
    parser.add_argument("--image", required=True, type=Path, help="Path to reference image")
    parser.add_argument("--out", required=True, type=Path, help="Output path for .npy embedding")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=10,
        help="Max faces to consider in reference image (default: 10)",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="face_encodings num_jitters (default: 1; higher = more accurate, slower)",
    )
    parser.add_argument(
        "--pad-fraction",
        type=float,
        default=0.2,
        help="Face crop padding as fraction of bbox (default: 0.2)",
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1

    img_bgr = load_image(image_path)
    if img_bgr is None:
        print(f"Error: could not load image: {image_path}", file=sys.stderr)
        return 1

    rgb = bgr_to_rgb(img_bgr)
    boxes = detect_faces(
        rgb,
        min_detection_confidence=args.min_detection_confidence,
        max_faces=args.max_faces,
    )
    if not boxes:
        print("Error: no face found in reference image.", file=sys.stderr)
        return 1

    # Pick largest face
    best_box = get_largest_face(boxes)
    assert best_box is not None
    face_crop = crop_face_region(rgb, best_box, pad_fraction=args.pad_fraction)
    embedder = FaceRecognitionEmbedder(num_jitters=args.num_jitters)
    embedding = embedder.embed_face(face_crop)
    embedding = np.asarray(embedding, dtype=np.float32)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embedding)
    print(f"Enrolled '{args.name}' from {image_path}")
    print(f"  Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"  Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
