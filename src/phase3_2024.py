"""
phase3_2024.py
--------------
Pull 2024 Koramangala Sentinel-2 composite for change detection.
"""

import sys
sys.path.append('/home/bharathd7900/geovision/src')

import ee
from geo_ingest import init_earth_engine, get_sentinel2_composite, export_to_gcs

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT   = "weighty-media-416305"
GCS_BUCKET    = "geovision-data"

REGION_NAME   = 'koramangala'
LON, LAT      = 77.6245, 12.9352
BUFFER_KM     = 1
START_2024    = "2024-01-01"
END_2024      = "2024-05-31"
MAX_CLOUD_PCT = 15

# ── Initialize ────────────────────────────────────────────────────────────────
print("Initializing Earth Engine...")
init_earth_engine(GCP_PROJECT)

# ── Pull 2024 composite ───────────────────────────────────────────────────────
print("Pulling 2024 Koramangala composite...")
composite_2024, region = get_sentinel2_composite(
    region_name=REGION_NAME,
    lon=LON,
    lat=LAT,
    buffer_km=BUFFER_KM,
    start_date=START_2024,
    end_date=END_2024,
    max_cloud_pct=MAX_CLOUD_PCT,
)
print("2024 composite created!")

# ── Export to GCS ─────────────────────────────────────────────────────────────
print("Exporting to GCS...")
task = export_to_gcs(
    image=composite_2024,
    region=region,
    filename="koramangala_s2_2024",
    bucket=GCS_BUCKET,
    gcs_prefix="geovision/phase3",
)

print(f"Task ID: {task.id}")
print("Check status:")
print(f"  python -c \"import ee; ee.Initialize(project='{GCP_PROJECT}'); print(ee.data.getTaskStatus('{task.id}')[0]['state'])\"")