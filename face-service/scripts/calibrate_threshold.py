#!/usr/bin/env python3
"""Suggest a tolerance threshold from a small set of positive and negative image paths.

Usage:
  python scripts/calibrate_threshold.py --ref known/person1.npy --positives data/positives.txt --negatives data/negatives.txt
  (positives.txt / negatives.txt: one image path per line)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.detector_mediapipe import detect_faces
from app.embedder import FaceRecognitionEmbedder, euclidean_distance
from app.io_image import bgr_to_rgb, crop_face_region, load_image
from app.matcher import match_image
from app.types import FaceResult


def load_paths_file(path: Path) -> list[Path]:
    """One path per line; skip empty and comments."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Path(line))
    return out


def get_best_distance_for_image(
    image_path: Path,
    ref_embedding: np.ndarray,
    embedder: FaceRecognitionEmbedder,
    min_detection_confidence: float = 0.5,
    max_faces: int = 10,
    pad_fraction: float = 0.2,
) -> tuple[float | None, int]:
    """Return (min distance to ref over all faces, num_faces). None if no faces."""
    img_bgr = load_image(image_path)
    if img_bgr is None:
        return None, 0
    rgb = bgr_to_rgb(img_bgr)
    boxes = detect_faces(rgb, min_detection_confidence=min_detection_confidence, max_faces=max_faces)
    if not boxes:
        return None, 0
    distances: list[float] = []
    for box in boxes:
        crop = crop_face_region(rgb, box, pad_fraction=pad_fraction)
        try:
            emb = embedder.embed_face(crop)
            d = euclidean_distance(ref_embedding, emb)
            distances.append(d)
        except Exception:
            continue
    if not distances:
        return None, len(boxes)
    return min(distances), len(boxes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suggest tolerance from positive/negative image lists."
    )
    parser.add_argument("--ref", required=True, type=Path, help="Path to .npy reference embedding")
    parser.add_argument(
        "--positives",
        type=Path,
        default=None,
        help="File with one image path per line (should match person)",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        default=None,
        help="File with one image path per line (should NOT match person)",
    )
    parser.add_argument("--num-jitters", type=int, default=1, help="face_encodings num_jitters")
    args = parser.parse_args()

    if not args.ref.exists():
        print(f"Error: reference not found: {args.ref}", file=sys.stderr)
        return 1

    ref_embedding = np.load(args.ref)
    if ref_embedding.ndim != 1:
        ref_embedding = ref_embedding.flatten()
    ref_embedding = np.asarray(ref_embedding, dtype=np.float64)

    embedder = FaceRecognitionEmbedder(num_jitters=args.num_jitters)

    pos_distances: list[float] = []
    if args.positives:
        for p in load_paths_file(args.positives):
            if not p.exists():
                print(f"Warning: positive path not found: {p}", file=sys.stderr)
                continue
            d, n = get_best_distance_for_image(p, ref_embedding, embedder)
            if d is not None:
                pos_distances.append(d)
            else:
                print(f"Warning: no face in positive image: {p}", file=sys.stderr)

    neg_distances: list[float] = []
    if args.negatives:
        for p in load_paths_file(args.negatives):
            if not p.exists():
                print(f"Warning: negative path not found: {p}", file=sys.stderr)
                continue
            d, n = get_best_distance_for_image(p, ref_embedding, embedder)
            if d is not None:
                neg_distances.append(d)
            # no face = true negative, skip

    print("Calibration results:")
    if pos_distances:
        print(f"  Positives (n={len(pos_distances)}): min={min(pos_distances):.4f}, max={max(pos_distances):.4f}, mean={np.mean(pos_distances):.4f}")
    else:
        print("  Positives: (none)")
    if neg_distances:
        print(f"  Negatives (n={len(neg_distances)}): min={min(neg_distances):.4f}, max={max(neg_distances):.4f}, mean={np.mean(neg_distances):.4f}")
    else:
        print("  Negatives: (none)")

    if pos_distances and neg_distances:
        max_pos = max(pos_distances)
        min_neg = min(neg_distances)
        gap = min_neg - max_pos
        suggested = max_pos + 0.5 * gap if gap > 0 else max_pos
        print(f"\n  Suggested tolerance: {suggested:.4f} (midpoint between max_positive and min_negative)")
        print("  Use --tolerance on find_person.py; calibrate for your dataset.")
    elif pos_distances:
        print(f"\n  Suggested tolerance: at least {max(pos_distances):.4f} to include all positives.")
    else:
        print("\n  Add positive samples to suggest a threshold.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
