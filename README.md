# Python services

This folder contains all Python-based services for Smart Event Gallery (SEG). The frontend and Next.js API call these services over HTTP.

## Contents

| Service | Description | Docs |
|--------|-------------|------|
| **face-service** | Face detection, matching, and embedding (MediaPipe + face_recognition). FastAPI on port 8000. | [face-service/README.md](face-service/README.md) |

## Running Python services

- **face-service**: See [face-service/README.md](face-service/README.md) for setup, run, and API details.
- Start the service before using face features from the frontend (e.g. `uvicorn app.main:app --host 0.0.0.0 --port 8000` from `python/face-service/`).

## Project layout

```
python/
├── README.md           (this file)
└── face-service/       FastAPI app + scripts for face detection/matching
    ├── README.md
    ├── app/            FastAPI app (main.py, detector, embedder, etc.)
    ├── scripts/        enroll.py, find_person.py, calibrate_threshold.py
    ├── tests/
    ├── requirements.txt
    └── Dockerfile
```

## Main repo docs

- [Backend structure](../docs/BACKEND_STRUCTURE.md) — API endpoints including Face Service
- [App flow](../docs/APP_FLOW.md) — How the frontend talks to the Face Service
- [Tech stack](../docs/TECH_STACK.md) — Python/FastAPI versions
