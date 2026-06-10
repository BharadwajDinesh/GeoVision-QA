# ── CUDA base image (GPU-enabled) ─────────────────────────────────────────────
# Matches the VM's driver (CUDA 12.4). Includes CUDA runtime libraries.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# System dependencies — Python + GDAL + build tools for geospatial wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential \
    g++ \
    gcc \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt .

# 1. Upgrade pip + build helpers
# 2. Install GPU (CUDA 12.1) torch — bitsandbytes needs the CUDA build
# 3. Install pip GDAL pinned to the system GDAL version
# 4. Install the rest of requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && export GDAL_VERSION=$(gdal-config --version) \
    && pip install --no-cache-dir "GDAL==${GDAL_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime config ────────────────────────────────────────────────────────────
ENV GCS_BUCKET=geovision-data \
    MAX_IMAGE_MB=20 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
