FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    pkg-config \
    python3-dev \
    libegl1 \
    libgles2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cap OpenBLAS to 1 thread per worker so N workers use exactly N vCPUs cleanly.
# Without this, each worker tries to grab all cores → contention on multi-worker setups.
ENV OPENBLAS_NUM_THREADS=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/mixed out known

# Expose FastAPI port
EXPOSE 8000

# Start application
# 4 worker processes — 1 vCPU each (OPENBLAS_NUM_THREADS=1), ~500 MB RSS each (~2 GB total).
# Worker count is controlled by FACE_SERVICE_WORKERS env var (default 4).
# Must match WORKER_CONCURRENCY in Node workers to avoid fetch-failed errors under load.
# Low-memory fallback: uvicorn app.main:app --host 0.0.0.0 --port 8000  (~512 MB RSS)
CMD gunicorn app.main:app -w ${FACE_SERVICE_WORKERS:-4} -k uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
