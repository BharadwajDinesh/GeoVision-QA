"""
phase3_run.py
-------------
Phase 3: Change Detection Pipeline
1. Pull 2015 Sentinel-2 composite from Earth Engine
2. Export to GCS
3. Run ChangeFormer on 2015 vs 2024 tiles
"""

import sys
sys.path.append('/home/bharathd7900/geovision/src')

import ee
from geo_ingest import init_earth_engine, get_sentinel2_composite, export_to_gcs

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT   = "weighty-media-416305"
GCS_BUCKET    = "geovision-data"

# Same coordinates as Phase 1 — critical for alignment
REGION_NAME   = 'Koramangala'
LON, LAT      = 77.626579, 12.934533
BUFFER_KM     = 1

# 2015 date range
# Widen the date range to get more scenes
START_2015    = "2015-06-23"   # Sentinel-2 first image
END_2015      = "2016-12-31"   # extend to end of 2016
MAX_CLOUD_PCT = 30              # also relax cloud threshold

# ── Step 1: Initialize Earth Engine ──────────────────────────────────────────
print("="*50)
print("STEP 1: Initializing Earth Engine")
print("="*50)
init_earth_engine(GCP_PROJECT)

# ── Step 2: Pull 2015 composite ───────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: Pulling 2015 Sentinel-2 composite")
print("="*50)
composite_2015, region = get_sentinel2_composite(
    region_name=REGION_NAME,
    lon=LON,
    lat=LAT,
    buffer_km=BUFFER_KM,
    start_date=START_2015,
    end_date=END_2015,
    max_cloud_pct=MAX_CLOUD_PCT,
)
print("2015 composite created!")

# ── Step 3: Export to GCS ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Exporting to GCS")
print("="*50)
task = export_to_gcs(
    image=composite_2015,
    region=region,
    filename="koramangala_s2_2015",
    bucket=GCS_BUCKET,
    gcs_prefix="geovision/phase3",
)

print(f"Export task started — task ID: {task.id}")