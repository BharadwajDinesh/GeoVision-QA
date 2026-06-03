"""
osm_labels.py
-------------
Reads OpenStreetMap features from a local GeoJSON file and
rasterizes them into a pixel-level segmentation mask.
Saves the result directly to GCS.

Classes:
    0 = background
    1 = building
    2 = road
    3 = vegetation
    4 = water

Usage:
    from osm_labels import OSMLabeler
    labeler = OSMLabeler()
    full_mask = labeler.build_mask()
    labeler.save_mask_to_gcs(full_mask)
"""

import io
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from google.cloud import storage


# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET    = "geovision-data"
GCS_TIF_PATH  = "geovision/phase1/bengaluru_s2_composite_2024.tif"
GCS_MASK_PATH = "geovision/phase2/full_mask.npy"

# ── Local GeoJSON path ────────────────────────────────────────────────────────
GEOJSON_PATH  = "/home/bharathd7900/geovision/data/raw/koramangala.geojson"


# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_MAP = {
    'background': 0,
    'building':   1,
    'road':       2,
    'vegetation': 3,
    'water':      4,
}

# OSM tag to class mapping
TAG_CLASS_MAP = {
    # buildings
    'building':   'building',
    # roads
    'highway':    'road',
    # vegetation
    'landuse':    'vegetation',
    'natural':    'vegetation',
    # water
    'waterway':   'water',
}

WATER_TAGS    = {'natural': ['water'], 'waterway': ['river', 'stream', 'canal', 'drain'], 'landuse': ['reservoir', 'basin']}
VEG_TAGS      = {'landuse': ['forest', 'grass', 'meadow', 'recreation_ground', 'village_green', 'allotments', 'orchard'], 'natural': ['wood', 'scrub', 'grassland', 'heath']}
ROAD_TAGS     = {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified']}

# Rasterization priority — last drawn wins
LAYER_ORDER = ['vegetation', 'water', 'road', 'building']


class OSMLabeler:
    """
    Reads OSM features from a local GeoJSON file and rasterizes
    them into a segmentation mask matching the GeoTIFF dimensions.
    Saves mask to GCS.
    """

    def __init__(self, geojson_path: str = GEOJSON_PATH):
        self.geojson_path = geojson_path
        self.client       = storage.Client()
        self.bucket       = self.client.bucket(GCS_BUCKET)

        # Read raster metadata from GCS
        print(f"Reading raster metadata from gs://{GCS_BUCKET}/{GCS_TIF_PATH} ...")
        blob = self.bucket.blob(GCS_TIF_PATH)
        data = blob.download_as_bytes()
        with rasterio.open(io.BytesIO(data)) as src:
            self.crs       = src.crs
            self.height    = src.height
            self.width     = src.width
            self.transform = src.transform
            self.bounds    = src.bounds

        print(f"Raster size : {self.width} x {self.height} px")
        print(f"CRS         : {self.crs}")

        # Load GeoJSON once
        print(f"\nLoading GeoJSON from {geojson_path} ...")
        self.gdf = gpd.read_file(geojson_path)
        print(f"Total features loaded: {len(self.gdf)}")
        print(f"Columns: {list(self.gdf.columns)}")

    def _filter_class(self, class_name: str) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame for a specific class based on OSM tags."""
        gdf = self.gdf

        if class_name == 'building':
            mask = gdf.get('building', gpd.pd.Series(dtype=str)).notna()
            filtered = gdf[mask]

        elif class_name == 'road':
            if 'highway' in gdf.columns:
                valid = ROAD_TAGS['highway']
                mask  = gdf['highway'].isin(valid)
                filtered = gdf[mask]
            else:
                filtered = gpd.GeoDataFrame(geometry=[])

        elif class_name == 'vegetation':
            masks = []
            if 'landuse' in gdf.columns:
                masks.append(gdf['landuse'].isin(VEG_TAGS['landuse']))
            if 'natural' in gdf.columns:
                masks.append(gdf['natural'].isin(VEG_TAGS['natural']))
            if masks:
                combined = masks[0]
                for m in masks[1:]:
                    combined = combined | m
                filtered = gdf[combined]
            else:
                filtered = gpd.GeoDataFrame(geometry=[])

        elif class_name == 'water':
            masks = []
            for tag, values in WATER_TAGS.items():
                if tag in gdf.columns:
                    masks.append(gdf[tag].isin(values))
            if masks:
                combined = masks[0]
                for m in masks[1:]:
                    combined = combined | m
                filtered = gdf[combined]
            else:
                filtered = gpd.GeoDataFrame(geometry=[])

        else:
            filtered = gpd.GeoDataFrame(geometry=[])

        print(f"  {class_name:12s}: {len(filtered):6d} features")
        return filtered

    def _rasterize_layer(self, gdf: gpd.GeoDataFrame, class_id: int) -> np.ndarray:
        """Burn a GeoDataFrame into a 2D uint8 array."""
        out = np.zeros((self.height, self.width), dtype=np.uint8)

        if gdf is None or len(gdf) == 0:
            return out

        try:
            gdf = gdf.to_crs(self.crs)
        except Exception as e:
            print(f"    CRS reprojection failed: {e}")
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
            all_touched=True,
        )
        return burned

    def build_mask(self) -> np.ndarray:
        """
        Build full-scene segmentation mask from local GeoJSON.

        Returns:
            2D uint8 numpy array (height, width) with class labels 0-4
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        print("\nFiltering features by class...")
        layers = {}
        for class_name in LAYER_ORDER:
            layers[class_name] = self._filter_class(class_name)

        print("\nRasterizing layers...")
        for class_name in LAYER_ORDER:
            class_id = CLASS_MAP[class_name]
            layer    = self._rasterize_layer(layers[class_name], class_id)
            mask     = np.where(layer > 0, layer, mask)
            pct      = (mask == class_id).sum() / mask.size * 100
            print(f"  {class_name:12s} ({class_id}): {pct:.1f}% of scene")

        print("\nFinal mask distribution:")
        total = mask.size
        for name, cid in CLASS_MAP.items():
            pct = (mask == cid).sum() / total * 100
            print(f"  {name:12s} ({cid}): {pct:.1f}%")

        return mask

    # ── GCS I/O ───────────────────────────────────────────────────────────────

    def save_mask_to_gcs(self, mask: np.ndarray) -> None:
        """Upload full mask to GCS."""
        buf = io.BytesIO()
        np.save(buf, mask)
        buf.seek(0)
        blob = self.bucket.blob(GCS_MASK_PATH)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        print(f"\nMask saved -> gs://{GCS_BUCKET}/{GCS_MASK_PATH}")
        print(f"  Shape          : {mask.shape}")
        print(f"  Unique classes : {np.unique(mask)}")

    def load_mask_from_gcs(self) -> np.ndarray:
        """Download previously saved mask from GCS."""
        print(f"Loading mask from gs://{GCS_BUCKET}/{GCS_MASK_PATH} ...")
        blob = self.bucket.blob(GCS_MASK_PATH)
        data = blob.download_as_bytes()
        mask = np.load(io.BytesIO(data))
        print(f"  Mask loaded: shape={mask.shape}, classes={np.unique(mask)}")
        return mask

    def mask_exists_in_gcs(self) -> bool:
        """Check if mask already exists in GCS."""
        blob = self.bucket.blob(GCS_MASK_PATH)
        return blob.exists()
