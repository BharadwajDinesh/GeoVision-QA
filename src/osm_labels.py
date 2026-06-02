"""
osm_labels.py
-------------
Downloads OpenStreetMap features for a geographic bounding box,
rasterizes them into a pixel-level segmentation mask, and saves
the result directly to GCS.

Classes:
    0 = background
    1 = building
    2 = road
    3 = vegetation
    4 = water

Usage:
    from osm_labels import OSMLabeler
    labeler = OSMLabeler()                   # reads GeoTIFF meta from GCS
    full_mask = labeler.build_mask()         # downloads OSM + rasterizes
    labeler.save_mask_to_gcs(full_mask)      # uploads mask to GCS
"""

import io
import numpy as np
import geopandas as gpd
import osmnx as ox
import rasterio
import rasterio.features
from rasterio.transform import Affine
from google.cloud import storage


# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET      = "geovision-data"
GCS_TIF_PATH    = "geovision/phase1/bengaluru_s2_composite_2024.tif"
GCS_MASK_PATH   = "geovision/phase2/full_mask.npy"


# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_MAP = {
    'background': 0,
    'building':   1,
    'road':       2,
    'vegetation': 3,
    'water':      4,
}

# OSM tags for each class
OSM_TAGS = {
    'building': {
        'building': True,
    },
    'road': {
        'highway': [
            'motorway', 'trunk', 'primary', 'secondary',
            'tertiary', 'residential', 'unclassified',
        ],
    },
    'vegetation': {
        'landuse': [
            'forest', 'grass', 'meadow', 'recreation_ground',
            'village_green', 'allotments', 'orchard', 'vineyard',
        ],
        'natural': ['wood', 'scrub', 'grassland', 'heath'],
    },
    'water': {
        'natural':  ['water'],
        'waterway': ['river', 'stream', 'canal', 'drain'],
        'landuse':  ['reservoir', 'basin'],
    },
}

# Rasterization priority — last drawn wins on overlap
# Buildings most specific → drawn last → highest priority
LAYER_ORDER = ['vegetation', 'water', 'road', 'building']


class OSMLabeler:
    """
    Downloads and rasterizes OSM features into a segmentation mask.
    Reads GeoTIFF metadata from GCS. Saves full mask back to GCS.
    """

    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(GCS_BUCKET)

        # Read raster metadata from GCS (no need to load full image)
        print(f"Reading raster metadata from gs://{GCS_BUCKET}/{GCS_TIF_PATH} ...")
        blob  = self.bucket.blob(GCS_TIF_PATH)
        data  = blob.download_as_bytes()
        with rasterio.open(io.BytesIO(data)) as src:
            self.bounds    = src.bounds
            self.crs       = src.crs
            self.height    = src.height
            self.width     = src.width
            self.transform = src.transform

        # bbox in (south, west, north, east) format for osmnx
        self.bbox = (
            self.bounds.bottom,
            self.bounds.left,
            self.bounds.top,
            self.bounds.right,
        )

        self._gdfs = {}   # cache of downloaded GeoDataFrames
        print(f"  Bounds : {self.bounds}")
        print(f"  Size   : {self.width} x {self.height} px")
        print(f"  CRS    : {self.crs}")

    # ── Download ──────────────────────────────────────────────────────────────

    def download(self, class_name: str) -> gpd.GeoDataFrame:
        """Download OSM features for a single class."""
        if class_name in self._gdfs:
            return self._gdfs[class_name]

        tags = OSM_TAGS.get(class_name, {})
        try:
            gdf = ox.features_from_bbox(bbox=self.bbox, tags=tags)
            print(f"  {class_name:12s}: {len(gdf):6d} features downloaded")
        except Exception as e:
            print(f"  {class_name:12s}: 0 features ({type(e).__name__})")
            gdf = gpd.GeoDataFrame(geometry=[])

        self._gdfs[class_name] = gdf
        return gdf

    def download_all(self) -> None:
        """Download all OSM classes."""
        print("\nDownloading OSM features...")
        for class_name in LAYER_ORDER:
            self.download(class_name)
        print("Download complete.")

    # ── Rasterize ─────────────────────────────────────────────────────────────

    def _rasterize_layer(self, gdf: gpd.GeoDataFrame, class_id: int) -> np.ndarray:
        """Burn a GeoDataFrame into a 2D uint8 array."""
        out = np.zeros((self.height, self.width), dtype=np.uint8)

        if gdf is None or len(gdf) == 0:
            return out

        # Reproject to match target raster CRS
        try:
            gdf = gdf.to_crs(self.crs)
        except Exception:
            return out

        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return out

        burned = rasterio.features.rasterize(
            [(geom, class_id) for geom in geoms],
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,   # critical for thin features like roads
        )
        return burned

    def build_mask(self) -> np.ndarray:
        """
        Build the full-scene segmentation mask.
        Downloads OSM data if not already cached.

        Returns:
            2D uint8 numpy array of shape (height, width)
            with values in range [0, 4]
        """
        if not self._gdfs:
            self.download_all()

        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        print("\nRasterizing layers...")
        for class_name in LAYER_ORDER:
            gdf      = self._gdfs.get(class_name, gpd.GeoDataFrame())
            class_id = CLASS_MAP[class_name]
            layer    = self._rasterize_layer(gdf, class_id)
            mask     = np.where(layer > 0, layer, mask)
            pct      = (mask == class_id).sum() / mask.size * 100
            print(f"  {class_name:12s} ({class_id}): {pct:.1f}% of scene")

        # Print final class distribution
        print("\nFinal mask distribution:")
        total = mask.size
        for name, cid in CLASS_MAP.items():
            pct = (mask == cid).sum() / total * 100
            print(f"  {name:12s} ({cid}): {pct:.1f}%")

        return mask

    # ── GCS I/O ───────────────────────────────────────────────────────────────

    def save_mask_to_gcs(self, mask: np.ndarray) -> None:
        """Upload the full mask numpy array to GCS."""
        buf = io.BytesIO()
        np.save(buf, mask)
        buf.seek(0)
        blob = self.bucket.blob(GCS_MASK_PATH)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        print(f"\nMask saved → gs://{GCS_BUCKET}/{GCS_MASK_PATH}")
        print(f"  Shape          : {mask.shape}")
        print(f"  Unique classes : {np.unique(mask)}")

    def load_mask_from_gcs(self) -> np.ndarray:
        """Download a previously saved mask from GCS."""
        print(f"Loading mask from gs://{GCS_BUCKET}/{GCS_MASK_PATH} ...")
        blob = self.bucket.blob(GCS_MASK_PATH)
        data = blob.download_as_bytes()
        mask = np.load(io.BytesIO(data))
        print(f"  Mask loaded: shape={mask.shape}, classes={np.unique(mask)}")
        return mask

    def mask_exists_in_gcs(self) -> bool:
        """Check if a full mask already exists in GCS."""
        blob = self.bucket.blob(GCS_MASK_PATH)
        return blob.exists()
