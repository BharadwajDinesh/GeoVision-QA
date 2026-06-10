# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies — GDAL + build tools required to compile the GDAL,
# rasterio, and shapely Python wheels from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# GDAL needs these so the Python bindings compile against the system GDAL.
# The pip "GDAL" version MUST match the system gdal-bin version, so we read it
# dynamically and pin pip's GDAL to it at build time.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt .

# 1. Upgrade pip + install build helpers
# 2. Install CPU-only torch first (smaller than default CUDA build)
# 3. Install pip's GDAL pinned to the system GDAL version (avoids version clash)
# 4. Install the rest of requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
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

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
