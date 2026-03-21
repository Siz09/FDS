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
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def process_single_image(
    image_path: Path,
    ref_embedding: np.ndarray,
    min_detection_confidence: float,
    num_jitters: int,
    max_faces: int,
    pad_fraction: float,
    tolerance: float,
) -> dict:
    """Worker function for parallel processing."""
    img_bgr = load_image(image_path)
    if img_bgr is None:
        return {"path": str(image_path), "error": "Could not load image"}

    rgb = bgr_to_rgb(img_bgr)    # 1. Detection
    all_boxes = detect_faces(
        rgb,
        min_detection_confidence=min_detection_confidence,
        max_faces=max_faces,
    )
    if not all_boxes:
        return {
            "name": image_path.name,
            "path": str(image_path),
            "matched": False,
            "best_distance": None,
            "num_faces": 0,
        }

    # 2. Main Subject Filtering (Surgically Calibrated)
    # Background faces in dense crowds often cause false positives. 
    # We ignore faces that are < 5% of the area of the largest face.
    # (5% is enough to keep everyone in a group photo but kill background spectators)
    largest_area = max(b.area for b in all_boxes)
    main_boxes = [b for b in all_boxes if b.area >= (largest_area * 0.05)]
    
    # Also ignore extremely small faces (< 0.5% of total image area)
    total_area = rgb.shape[0] * rgb.shape[1]
    main_boxes = [b for b in main_boxes if b.area >= (total_area * 0.005)]

    # Limit to Top 8 largest faces to ensure group photos work but crowds are filtered
    main_boxes = sorted(main_boxes, key=lambda b: b.area, reverse=True)[:8]

    if not main_boxes:
        return {
            "name": image_path.name,
            "path": str(image_path),
            "matched": False,
            "best_distance": None,
            "num_faces": 0,
        }
    
    face_results: list[FaceResult] = []
    # Embedder must be initialized inside the worker for ProcessPool
    embedder = FaceRecognitionEmbedder(num_jitters=num_jitters, model="large")
    
    for box in main_boxes: # Optimized focus
        crop = crop_face_region(rgb, box, pad_fraction=pad_fraction)
        try:
            emb = embedder.embed_face(crop)
            face_results.append(FaceResult(bbox=box, embedding=emb, confidence=0.0))
        except Exception:
            continue

    if not face_results:
        return {
            "name": image_path.name,
            "path": str(image_path),
            "matched": False,
            "best_distance": None,
            "num_faces": 0,
        }

    result = match_image(ref_embedding, face_results, tolerance)
    return {
        "name": image_path.name,
        "path": str(image_path),
        "matched": result.matched,
        "best_distance": result.best_distance,
        "num_faces": result.num_faces,
    }


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

    matches: list[dict] = []
    report_entries: list[dict] = []
    
    # Run in parallel using 4 CPU cores (balanced for your 16GB RAM)
    print(f"Processing {len(image_paths)} images in parallel (using 4 cores)...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_single_image,
                path,
                ref_embedding,
                args.min_detection_confidence,
                args.num_jitters,
                args.max_faces,
                args.pad_fraction,
                args.tolerance,
            )
            for path in image_paths
        ]
        
        for future in as_completed(futures):
            res = future.result()
            if "error" in res:
                print(f"Warning: {res['error']} for {res['path']}", file=sys.stderr)
                continue
            
            report_entries.append(res)
            if res["matched"]:
                matches.append(res)
                print(f"MATCH\tname={res['name']}\tbest_distance={res['best_distance']:.4f}\tfaces={res['num_faces']}")

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
