"""
geo_ingest.py
-------------
Pulls Sentinel-2 imagery from Google Earth Engine, applies cloud masking,
and exports GeoTIFF tiles to a local directory.

Phase 1 of the GeoVision QA pipeline.
"""

import ee
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import requests

# ── Sentinel-2 band reference ────────────────────────────────────────────────
# We work with the L2A (surface reflectance) product.
# Bands are stored as uint16 scaled by 10000, so divide by 10000 to get [0,1].

S2_BANDS = {
    "B2":  "Blue        (490nm,  10m)",
    "B3":  "Green       (560nm,  10m)",
    "B4":  "Red         (665nm,  10m)",
    "B5":  "Red Edge 1  (705nm,  20m)",
    "B6":  "Red Edge 2  (740nm,  20m)",
    "B7":  "Red Edge 3  (783nm,  20m)",
    "B8":  "NIR         (842nm,  10m)",
    "B8A": "NIR narrow  (865nm,  20m)",
    "B11": "SWIR-1      (1610nm, 20m)",
    "B12": "SWIR-2      (2190nm, 20m)",
    "QA60": "Cloud mask  (60m)",
}

DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # RGB + NIR + SWIR


# ── Auth ─────────────────────────────────────────────────────────────────────

def init_earth_engine(project_id: str) -> None:
    """
    Authenticate and initialise the Earth Engine Python API.

    On a GCP VM with a service account that has Earth Engine access,
    authentication is automatic. On a fresh machine, this triggers
    a browser-based OAuth flow the first time.

    Args:
        project_id: Your GCP project ID (e.g. "my-gcp-project-123")
    """
    try:
        ee.Initialize(project=project_id)
        print(f"Earth Engine initialised. Project: {project_id}")
    except Exception:
        print("No credentials found — launching interactive auth...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("Auth complete.")


# ── Cloud masking ─────────────────────────────────────────────────────────────

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Apply the QA60 cloud bitmask to a Sentinel-2 L2A image.

    Bit 10 = opaque clouds
    Bit 11 = cirrus clouds

    Pixels flagged as either are masked out (set to nodata).
    """
    qa = image.select("QA60")
    cloud_bit_mask   = 1 << 10
    cirrus_bit_mask  = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
          .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    # Scale reflectance from uint16 → float [0, 1]
    return image.updateMask(mask).divide(10000)


# ── Image collection ──────────────────────────────────────────────────────────

def get_sentinel2_composite(
    region_name: str,
    lon: float,
    lat: float,
    buffer_km: float = 10.0,
    start_date: str = "2024-01-01",
    end_date: str   = "2024-06-30",
    max_cloud_pct: float = 20.0,
    bands: list[str] = DEFAULT_BANDS,
) -> tuple[ee.Image, ee.Geometry]:
    """
    Build a median cloud-free composite over a square region.

    A median composite takes the median pixel value across all valid
    (non-cloudy) observations in the date range. This is the standard
    baseline approach for getting a clean single image.

    Args:
        region_name:   Human-readable label (used in filenames)
        lon, lat:      Centre of the area of interest (decimal degrees)
        buffer_km:     Half-width of the square tile in kilometres
        start_date:    ISO date string "YYYY-MM-DD"
        end_date:      ISO date string "YYYY-MM-DD"
        max_cloud_pct: Filter out scenes with more cloud cover than this
        bands:         Sentinel-2 band names to include

    Returns:
        (composite image, geometry) — both EE objects ready for export
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_km * 1000).bounds()  # square bbox

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
          .map(mask_s2_clouds)
          .select(bands)
    )

    count = collection.size().getInfo()
    print(f"[{region_name}] Found {count} scenes after cloud filter "
          f"({start_date} → {end_date}, <{max_cloud_pct}% cloud)")

    if count == 0:
        raise ValueError(
            "No scenes found. Try widening the date range or increasing "
            "max_cloud_pct."
        )

    composite = collection.median().clip(region)
    return composite, region


# ── Export ────────────────────────────────────────────────────────────────────

def export_to_drive(
    image: ee.Image,
    region: ee.Geometry,
    filename: str,
    drive_folder: str = "geovision_phase1",
    scale_m: int = 10,
) -> ee.batch.Task:
    """
    Export an EE image to Google Drive as a GeoTIFF.

    Earth Engine can't write directly to local disk — it queues an async
    export task. You download the GeoTIFF from Drive afterward.

    Args:
        image:        EE Image to export
        region:       Export extent (EE Geometry)
        filename:     Output filename without extension
        drive_folder: Google Drive folder name (created automatically)
        scale_m:      Output pixel size in metres (10 = native S2 res)

    Returns:
        The submitted EE Task object (call .status() to poll progress)
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=drive_folder,
        fileNamePrefix=filename,
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()
    print(f"Export task submitted: '{filename}' → Drive/{drive_folder}/")
    print(f"Task ID: {task.id}")
    print("Poll with: task.status()  |  Monitor at https://code.earthengine.google.com/tasks")
    return task


def export_to_gcs(
    image: ee.Image,
    region: ee.Geometry,
    filename: str,
    bucket: str,
    gcs_prefix: str = "geovision/phase1",
    scale_m: int = 10,
) -> ee.batch.Task:
    """
    Export an EE image directly to a GCS bucket — preferred on GCP VMs.

    Args:
        image:      EE Image to export
        region:     Export extent
        filename:   Output filename without extension
        bucket:     GCS bucket name (no gs:// prefix)
        gcs_prefix: Folder path inside the bucket
        scale_m:    Output pixel size in metres

    Returns:
        The submitted EE Task object
    """
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=filename,
        bucket=bucket,
        fileNamePrefix=f"{gcs_prefix}/{filename}",
        region=region,
        scale=scale_m,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e9,
    )
    task.start()
    print(f"Export task submitted: gs://{bucket}/{gcs_prefix}/{filename}.tif")
    print(f"Task ID: {task.id}")
    return task


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: Bengaluru urban area
    # Replace with your GCP project ID
    GCP_PROJECT = "your-gcp-project-id"

    init_earth_engine(GCP_PROJECT)

    composite, region = get_sentinel2_composite(
        region_name="bengaluru",
        lon=77.5946,
        lat=12.9716,
        buffer_km=15,
        start_date="2024-01-01",
        end_date="2024-05-31",
        max_cloud_pct=15,
    )

    # Export to Drive (or swap for export_to_gcs if you have a bucket)
    task = export_to_drive(
        image=composite,
        region=region,
        filename="bengaluru_s2_composite_2024",
    )
