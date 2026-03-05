# Face Service

Python face detection and matching service for **Smart Event Gallery (SEG)**. This service lives in `python/face-service/` in the SEG repo. It provides a **FastAPI** HTTP API and CLI scripts for enrolling and finding a person in images.

## Architecture

*   **FastAPI app** (`app/main.py`): HTTP endpoints — `POST /detect-face`, `POST /match-face`, `GET /health`. Run with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
*   **Scripts**: `enroll.py`, `find_person.py`, `calibrate_threshold.py` for batch enrollment and search.
*   **Stack**: MediaPipe (detection), `face_recognition` / dlib (128-d embeddings), OpenCV, NumPy.

The SEG frontend and Next.js API call this service over HTTP when face features are used.

---

## Prerequisites

1.  **Python 3.10+** (3.11 recommended).
2.  **OS**: Windows, Linux, or macOS.
    *   **Windows**: Prefer `dlib-bin` to avoid compiling dlib; otherwise Visual C++ Build Tools may be required.

---

## Setup Guide

### 1. Backend Setup (this directory: `python/face-service/`)

Run all commands below from **`python/face-service/`** (repo root: `SEG/python/face-service/`).

**Windows:**

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Install dependencies
# Install dlib binary first to avoid compilation issues
pip install dlib-bin
# Install face_recognition without dependencies (to use the pre-installed dlib)
pip install face_recognition --no-deps
# Install remaining requirements
pip install -r requirements-windows.txt

# 4. Verify installation
pytest tests/ -v
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

**One-time Data Folder Setup:**
Create the necessary directories for storing images and data.

```bash
# Windows
mkdir data data\mixed known out

# Linux/macOS
mkdir -p data/mixed known out
```

### 2. Run the Face Service API

From `python/face-service/` with your venv activated:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **Health**: [http://localhost:8000/health](http://localhost:8000/health)
- **OpenAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)

The SEG frontend/API should be configured to call this base URL (e.g. `http://localhost:8000` or via `NEXT_PUBLIC_FACE_SERVICE_URL` / equivalent).

### 3. Frontend (SEG monorepo)

The main app is in the repo root. From the repo root:

```bash
cd apps/frontend-app
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Ensure the Face Service is running on port 8000 when using face features.

---

## Manual / Pure Script Usage

You can also run the Python scripts directly without the frontend.

**Enroll Person-1:**
```bash
python scripts/enroll.py --name person1 --image data/person1.jpg --out known/person1.npy
```

**Find Person-1 in a folder:**
```bash
python scripts/find_person.py --name person1 --ref known/person1.npy --folder data/mixed --tolerance 0.6
```

---

## Troubleshooting

*   **`dlib` installation fails**: On Windows, install `dlib-bin` first, or use CMake and Visual Studio C++ build tools.
*   **Face features not working in the app**: Ensure the Face Service is running (`uvicorn app.main:app --host 0.0.0.0 --port 8000` from `python/face-service/`) and the frontend/API is configured to use its URL.

---

## Repo docs

*   [Python folder README](../README.md) — overview of `python/`
*   [Backend structure](../../docs/BACKEND_STRUCTURE.md) — Face Service API details
*   [App flow](../../docs/APP_FLOW.md) — how the frontend uses the Face Service
