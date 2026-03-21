# Working Snapshot — 2026-03-21

## Verified Results
- 5/5 true positives matched (IMG_7439, IMG_7440, IMG_7441, IMG_7446, IMG_7507)
- 0 false positives
- SEG `FACE_MATCH_THRESHOLD = 0.535` (packages/db/src/index.ts)

## Key Settings
- `/embed-face` tolerance: dual-layer filter (15% relative + 0.5% absolute area) + Mirror TTA
- `/match-face` base tolerance: 0.55 with dynamic penalty (area_ratio * 0.25)
- dlib "large" model, num_jitters=1 (TTA handles stability instead)
- Ensemble detection: MediaPipe (blaze_face_full_range) + HOG with IoU merging

## To restore if broken
Copy all .py files from this directory back to their original locations:
- main.py → app/main.py
- middleware.py → app/middleware.py
- detector_mediapipe.py → app/detector_mediapipe.py
- embedder.py → app/embedder.py
- vault.py → app/vault.py
- find_person.py → scripts/find_person.py
- Dockerfile → Dockerfile

Then rebuild Docker:
  cd SEG/infrastructure && docker compose up --build face-service -d
