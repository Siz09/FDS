#!/usr/bin/env python3
"""Find images in a folder that contain Person-1 (reference embedding).

Usage:
  python scripts/find_person.py --name person1 --ref known/person1.npy --folder data/mixed --tolerance 0.6
  python scripts/find_person.py --name person1 --ref known/person1.npy --folder data/mixed --tolerance 0.6 --report out/report.json
  python scripts/find_person.py ... --save-annotated out/annotated
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.detector_mediapipe import detect_faces
from app.embedder import FaceRecognitionEmbedder
from app.io_image import bgr_to_rgb, crop_face_region, load_image
from app.matcher import match_image
from app.types import FaceResult, FaceBox


# Image extensions to scan
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(folder: Path) -> list[Path]:
    """Return sorted list of image paths under folder (non-recursive by default)."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(
    image_path: Path,
    rgb: np.ndarray,
    embedder: FaceRecognitionEmbedder,
    min_detection_confidence: float,
    max_faces: int,
    pad_fraction: float,
) -> list[FaceResult]:
    """Detect faces, crop, embed; return list of FaceResult with embeddings."""
    boxes = detect_faces(
        rgb,
        min_detection_confidence=min_detection_confidence,
        max_faces=max_faces,
    )
    results: list[FaceResult] = []
    for box in boxes:
        crop = crop_face_region(rgb, box, pad_fraction=pad_fraction)
        try:
            emb = embedder.embed_face(crop)
            results.append(FaceResult(bbox=box, embedding=emb, confidence=0.0))
        except Exception:
            continue
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find images in folder that contain the person (reference embedding)."
    )
    parser.add_argument("--name", required=True, help="Label for the person (e.g. person1)")
    parser.add_argument("--ref", required=True, type=Path, help="Path to .npy reference embedding")
    parser.add_argument("--folder", required=True, type=Path, help="Folder of images to scan")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Max Euclidean distance to consider a match (default: 0.6; lower = stricter)",
    )
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
        help="Max faces per image (default: 10)",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="face_encodings num_jitters (default: 1)",
    )
    parser.add_argument(
        "--pad-fraction",
        type=float,
        default=0.2,
        help="Face crop padding (default: 0.2)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--save-annotated",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save matched images with bbox + distance label into DIR",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan folder recursively",
    )
    args = parser.parse_args()

    if not args.ref.exists():
        print(f"Error: reference embedding not found: {args.ref}", file=sys.stderr)
        return 1

    ref_embedding = np.load(args.ref)
    if ref_embedding.ndim != 1:
        ref_embedding = ref_embedding.flatten()
    ref_embedding = np.asarray(ref_embedding, dtype=np.float64)

    if not args.folder.is_dir():
        print(f"Error: folder not found: {args.folder}", file=sys.stderr)
        return 1

    if args.recursive:
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(args.folder.rglob(f"*{ext}"))
        image_paths = sorted(set(image_paths))
    else:
        image_paths = find_images(args.folder)

    if not image_paths:
        print(f"No images found in {args.folder}", file=sys.stderr)
        return 0

    embedder = FaceRecognitionEmbedder(num_jitters=args.num_jitters)
    matches: list[dict] = []
    report_entries: list[dict] = []
    annotated_dir = args.save_annotated
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img_bgr = load_image(path)
        if img_bgr is None:
            print(f"Warning: could not load image, skipping: {path}", file=sys.stderr)
            continue
        rgb = bgr_to_rgb(img_bgr)
        face_results = process_image(
            path,
            rgb,
            embedder,
            args.min_detection_confidence,
            args.max_faces,
            args.pad_fraction,
        )
        if not face_results:
            # No faces in image: skip (not a match)
            report_entries.append({
                "name": path.name,
                "path": str(path),
                "matched": False,
                "best_distance": None,
                "num_faces": 0,
            })
            continue

        result = match_image(ref_embedding, face_results, args.tolerance)
        report_entries.append({
            "name": path.name,
            "path": str(path),
            "matched": result.matched,
            "best_distance": result.best_distance,
            "num_faces": result.num_faces,
        })
        if result.matched:
            matches.append({
                "name": path.name,
                "path": str(path),
                "best_distance": result.best_distance,
                "num_faces": result.num_faces,
            })
            print(f"MATCH\tname={path.name}\tpath={path}\tbest_distance={result.best_distance:.4f}\tfaces={result.num_faces}")

            if annotated_dir:
                try:
                    import cv2
                    ann = img_bgr.copy()
                    for box in result.face_boxes:
                        cv2.rectangle(
                            ann,
                            (box.x, box.y),
                            (box.x + box.w, box.y + box.h),
                            (0, 255, 0),
                            2,
                        )
                    label = f"d={result.best_distance:.3f}"
                    cv2.putText(
                        ann,
                        label,
                        (result.face_boxes[0].x, result.face_boxes[0].y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    out_name = path.name
                    out_path = annotated_dir / out_name
                    cv2.imwrite(str(out_path), ann)
                except Exception as e:
                    print(f"Warning: could not save annotated image: {e}", file=sys.stderr)

    print()
    print(f"Summary: {len(matches)} match(es) out of {len(image_paths)} image(s) (tolerance={args.tolerance})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(
                {
                    "person": args.name,
                    "tolerance": args.tolerance,
                    "total_images": len(image_paths),
                    "matches": matches,
                    "all_results": report_entries,
                },
                f,
                indent=2,
            )
        print(f"Report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
