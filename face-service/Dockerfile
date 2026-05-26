FROM python:3.11-slim

# Install system dependencies.
# Removed: cmake, libopenblas-dev, liblapack-dev, libx11-dev, libgtk-3-dev,
#          gfortran — all were needed only for dlib compilation.
# Removed: libegl1, libgles2, libgl1 — MediaPipe EGL requirements, not needed for SCRFD.
# Kept: OpenCV runtime deps only.
RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + models directory.
# Required model files in models/ before building:
#   w600k_r100.onnx        (~260 MB) — glintr100 from antelopev2, ArcFace R100
#   2d106det.onnx           (~10 MB) — landmark detector from buffalo_l
#   scrfd_10g_bnkps.onnx    (~17 MB) — SCRFD-10GF face detector from antelopev2
# For Hetzner: scp the files there once, then rebuilds are instant — no network.
# Download: https://github.com/deepinsight/insightface/releases (antelopev2 + buffalo_l packs)
COPY . .

# Ensure runtime data directories exist
RUN mkdir -p data/mixed out known models

# Fail the build loudly if any model file is absent — better than a silent crash at startup.
RUN python -c "\
from pathlib import Path; \
p1 = Path('/app/models/w600k_r100.onnx'); \
assert p1.exists() and p1.stat().st_size > 250_000_000, \
    f'w600k_r100.onnx missing or too small at {p1} — place the ~260 MB ONNX file in models/ before building'; \
print(f'ArcFace R100 model present: {p1.stat().st_size // 1_000_000} MB', flush=True); \
p2 = Path('/app/models/2d106det.onnx'); \
assert p2.exists() and p2.stat().st_size > 5_000_000, \
    f'2d106det.onnx missing or too small at {p2} — extract it from buffalo_l.zip and place in models/ before building'; \
print(f'2d106det landmark model present: {p2.stat().st_size // 1_000_000} MB', flush=True); \
p3 = Path('/app/models/scrfd_10g_bnkps.onnx'); \
assert p3.exists() and p3.stat().st_size > 15_000_000, \
    f'scrfd_10g_bnkps.onnx missing at {p3} — download from InsightFace antelopev2 pack'; \
print(f'SCRFD detector model present: {p3.stat().st_size // 1_000_000} MB', flush=True)"

# Expose FastAPI port
EXPOSE 8000

# Start application
# 4 worker processes — 1 vCPU each, ~300 MB RSS each (~1.2 GB total, down from ~2 GB with dlib).
# Worker count is controlled by FACE_SERVICE_WORKERS env var (default 4).
CMD gunicorn app.main:app -w ${FACE_SERVICE_WORKERS:-4} -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
