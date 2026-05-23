FROM python:3.11-slim

# Install system dependencies.
# Removed: cmake, libopenblas-dev, liblapack-dev, libx11-dev, libgtk-3-dev,
#          gfortran — all were needed only for dlib compilation.
# Kept: OpenCV runtime deps + MediaPipe EGL requirements.
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
    libegl1 \
    libgles2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + models directory.
# models/w600k_r100.onnx (~260 MB) must exist on the build host at this path.
# Upgraded from r50 (174 MB) to r100 (ResNet-100/Glint360K) for better identity separation.
# For Hetzner: scp the file there once, then rebuilds are instant — no network.
#   scp python/face-service/models/w600k_r100.onnx hetzner:/app/models/
# Download: glintr100.onnx from https://github.com/deepinsight/insightface/releases (antelopev2 pack)
# For local dev: place glintr100.onnx renamed to w600k_r100.onnx in face-service/models/ manually.
COPY . .

# Ensure runtime data directories exist
RUN mkdir -p data/mixed out known models

# Fail the build loudly if the model file is absent — better than a silent crash at startup.
RUN python -c "\
from pathlib import Path; \
p1 = Path('/app/models/w600k_r100.onnx'); \
assert p1.exists() and p1.stat().st_size > 250_000_000, \
    f'w600k_r100.onnx missing or too small at {p1} — place the ~260 MB ONNX file in models/ before building'; \
print(f'ArcFace R100 model present: {p1.stat().st_size // 1_000_000} MB', flush=True); \
p2 = Path('/app/models/2d106det.onnx'); \
assert p2.exists() and p2.stat().st_size > 5_000_000, \
    f'2d106det.onnx missing or too small at {p2} — extract it from buffalo_l.zip and place in models/ before building'; \
print(f'2d106det landmark model present: {p2.stat().st_size // 1_000_000} MB', flush=True)"

# Expose FastAPI port
EXPOSE 8000

# Start application
# 4 worker processes — 1 vCPU each, ~300 MB RSS each (~1.2 GB total, down from ~2 GB with dlib).
# Worker count is controlled by FACE_SERVICE_WORKERS env var (default 4).
CMD gunicorn app.main:app -w ${FACE_SERVICE_WORKERS:-4} -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
