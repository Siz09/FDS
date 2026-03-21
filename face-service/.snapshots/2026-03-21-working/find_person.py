#!/usr/bin/env python3
"""Find images in a folder that contain the target person.
Optimized for SEG: High speed via Indexing (Vault), 8-core Parallelism, and Adaptive Jitters.
"""
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.detector_mediapipe import detect_faces
from app.embedder import FaceRecognitionEmbedder, euclidean_distance
from app.io_image import load_image, bgr_to_rgb, crop_face_region
from app.vault import FaceVault

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def process_single_image(image_path: Path, ref_embedding: list[float], num_jitters: int, tolerance: float, min_detection_confidence: float, params: dict):
    """Worker function for parallel processing with Vault (Indexing) support."""
    try:
        # 1. Check Vault (Instant Skip)
        vault = FaceVault()
        cached_faces = vault.get_image_data(str(image_path), params)
        
        face_results = []
        best_dist = 1.0
        matched = False
        
        if cached_faces:
            for face_data in cached_faces:
                dist = euclidean_distance(ref_embedding, face_data["embedding"])
                if dist < best_dist:
                    best_dist = dist
                if dist <= tolerance:
                    matched = True
            return {
                "name": image_path.name,
                "path": str(image_path),
                "matched": bool(matched),
                "best_distance": float(best_dist),
                "num_faces": len(cached_faces),
                "cached": True
            }

        # 2. Logic: Detection -> Filtering -> Embedding
        img_bgr = load_image(str(image_path))
        if img_bgr is None:
            return {"path": str(image_path), "error": "Could not load image"}

        img_rgb = bgr_to_rgb(img_bgr)
        all_boxes = detect_faces(img_rgb, min_detection_confidence=min_detection_confidence)
        if not all_boxes:
            return {"name": image_path.name, "path": str(image_path), "matched": False, "best_distance": 1.0, "num_faces": 0}

        # Subject Filtering: Google-Grade "Main Subject" Logic
        largest_area = max(b.area for b in all_boxes)
        # Tighter Filter: Keep faces > 15% of largest (eliminates background crowds)
        main_boxes = [b for b in all_boxes if b.area >= (largest_area * 0.15)]
        main_boxes = sorted(main_boxes, key=lambda b: b.area, reverse=True)[:8]

        embedder = FaceRecognitionEmbedder(num_jitters=num_jitters, model="large")
        for box in main_boxes:
            crop = crop_face_region(img_rgb, box)
            try:
                emb = embedder.embed_face(crop)
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                # Ratio of this face to the largest face in the image
                area_ratio = box.area / largest_area
                
                face_results.append({
                    "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h}, 
                    "embedding": emb_list,
                    "area_ratio": area_ratio
                })
                
                # Dynamic Tolerance Logic: Stricter for small background faces
                # If area_ratio=1.0 (Main Subject), tolerance is full.
                # If area_ratio=0.15 (Crowd), tolerance drops significantly (by up to 0.15).
                effective_tolerance = tolerance - (1.0 - area_ratio) * 0.25
                
                dist = euclidean_distance(ref_embedding, emb)
                if dist < best_dist:
                    best_dist = dist
                if dist <= effective_tolerance:
                    matched = True
            except Exception:
                continue

        # Save to Vault
        vault.add_image_data(str(image_path), face_results, params)
        
        return {
            "name": image_path.name,
            "path": str(image_path),
            "matched": bool(matched),
            "best_distance": float(best_dist),
            "num_faces": len(main_boxes),
            "cached": False
        }
    except Exception as e:
        return {"path": str(image_path), "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="SEG Optimized Face Matcher")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.55) # Synchronized "Normal" Threshold
    parser.add_argument("--jitters", "--num-jitters", type=int, default=5, dest="jitters")
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cores", type=int, default=8)
    args = parser.parse_args()

    if not args.ref.exists():
        print(f"Error: {args.ref} not found", file=sys.stderr)
        return 1

    # Load and normalize reference embedding
    ref_embedding = np.load(args.ref).flatten().astype(np.float64).tolist()

    image_paths = sorted([p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        print("No images found")
        return 0

    print(f"Searching for {args.name} in {len(image_paths)} images (Cores: {args.cores})...")
    start_time = time.time()
    report_entries = []
    matches = []

    # Important: Include THE PENALTY in the cache params so change in logic invalidates cache
    params = {
        "tolerance": args.tolerance,
        "jitters": args.jitters,
        "min_detection_confidence": args.min_detection_confidence,
        "dynamic_penalty": 0.25 # Current "Crowd-Crusher" constant
    }
        
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = [executor.submit(process_single_image, p, ref_embedding, args.jitters, args.tolerance, args.min_detection_confidence, params) for p in image_paths]
        for future in as_completed(futures):
            res = future.result()
            if "error" in res:
                print(f"Error in {res['path']}: {res['error']}")
                continue
            
            report_entries.append(res)
            if res["matched"]:
                matches.append(res)
                mode = "[CACHED]" if res.get("cached") else "[SCAN]"
                print(f"{mode} MATCH: {res['name']} (Dist: {res['best_distance']:.4f})")

    duration = time.time() - start_time
    print(f"\nDone in {duration:.2f}s. Found {len(matches)} matches.")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"person": args.name, "matches": matches, "all_results": report_entries}, f, indent=2)

if __name__ == "__main__":
    main()
