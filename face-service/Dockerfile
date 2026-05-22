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

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/mixed out known models

# Expose FastAPI port
EXPOSE 8000

# Start application
# 4 worker processes — 1 vCPU each, ~300 MB RSS each (~1.2 GB total, down from ~2 GB with dlib).
# Worker count is controlled by FACE_SERVICE_WORKERS env var (default 4).
CMD gunicorn app.main:app -w ${FACE_SERVICE_WORKERS:-4} -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
