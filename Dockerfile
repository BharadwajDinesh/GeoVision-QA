# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies for rasterio / GDAL / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: install Python deps ──────────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only torch first (smaller) — if you have a GPU VM, replace with
# the CUDA wheel: --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: copy source ──────────────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime config ────────────────────────────────────────────────────────────
# GCS_BUCKET       : name of the bucket holding all GeoVision artifacts
# GOOGLE_APPLICATION_CREDENTIALS : path to the service-account JSON (mounted at runtime)
ENV GCS_BUCKET=geovision-data \
    MAX_IMAGE_MB=20 \
    PORT=8080

EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --workers 1  : single worker — LLaVA is too large to fork
# --timeout 300: VQA inference can take ~60s on CPU; generous timeout
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
