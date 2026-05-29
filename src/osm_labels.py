"""
osm_labels.py
-------------
Downloads OpenStreetMap features for a geographic bounding box and
rasterizes them into a pixel-level segmentation mask.

Classes:
    0 = background
    1 = building
    2 = road
    3 = vegetation
    4 = water

Usage:
    from osm_labels import OSMLabeler
    labeler = OSMLabeler(bounds, crs, height, width, transform)
    full_mask = labeler.build_mask()
"""

import numpy as np
import geopandas as gpd
import osmnx as ox
from rasterio.features import rasterize
from rasterio.transform import Affine
from pathlib import Path


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
        'natural':   ['water'],
        'waterway':  ['river', 'stream', 'canal', 'drain'],
        'landuse':   ['reservoir', 'basin'],
    },
}

# Rasterization priority (higher = drawn on top, wins overlap)
# Buildings most specific → on top
LAYER_ORDER = ['vegetation', 'water', 'road', 'building']


class OSMLabeler:
    """
    Downloads and rasterizes OSM features into a segmentation mask.

    Args:
        bounds:    Rasterio BoundingBox (left, bottom, right, top)
        crs:       Coordinate reference system of the target raster
        height:    Height of the output mask in pixels
        width:     Width of the output mask in pixels
        transform: Rasterio Affine transform of the target raster
    """

    def __init__(
        self,
        bounds,
        crs,
        height: int,
        width: int,
        transform: Affine,
    ):
        self.bounds    = bounds
        self.crs       = crs
        self.height    = height
        self.width     = width
        self.transform = transform

        # bbox in (S, W, N, E) format for osmnx
        self.bbox = (bounds.bottom, bounds.left, bounds.top, bounds.right)

        self._gdfs = {}   # cache downloaded GeoDataFrames

    def download(self, class_name: str) -> gpd.GeoDataFrame:
        """
        Download OSM features for a single class.

        Args:
            class_name: One of 'building', 'road', 'vegetation', 'water'

        Returns:
            GeoDataFrame with geometry column, or empty GeoDataFrame on failure
        """
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
        """Download all OSM classes. Call this before build_mask()."""
        print("Downloading OSM features...")
        for class_name in LAYER_ORDER:
            self.download(class_name)
        print("Download complete.\n")

    def _rasterize_layer(
        self,
        gdf: gpd.GeoDataFrame,
        class_id: int,
    ) -> np.ndarray:
        """
        Burn a GeoDataFrame into a 2D numpy array.

        Args:
            gdf:      GeoDataFrame to rasterize
            class_id: Integer class label to burn

        Returns:
            2D uint8 array of shape (height, width)
        """
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

        burned = rasterize(
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
        Build the full-scene segmentation mask by rasterizing all layers.

        Layers are applied in LAYER_ORDER — buildings drawn last so they
        win any overlap with roads or vegetation.

        Returns:
            2D uint8 numpy array of shape (height, width)
            with values in range [0, 4]
        """
        if not self._gdfs:
            self.download_all()

        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        print("Rasterizing layers...")
        for class_name in LAYER_ORDER:
            gdf      = self._gdfs.get(class_name, gpd.GeoDataFrame())
            class_id = CLASS_MAP[class_name]
            layer    = self._rasterize_layer(gdf, class_id)
            # Overwrite mask where this layer has labels
            mask = np.where(layer > 0, layer, mask)
            pct  = (mask == class_id).sum() / mask.size * 100
            print(f"  {class_name:12s} ({class_id}): {pct:.1f}% of scene")

        # Print final class distribution
        print("\nFinal mask distribution:")
        total = mask.size
        for name, cid in CLASS_MAP.items():
            pct = (mask == cid).sum() / total * 100
            print(f"  {name:12s} ({cid}): {pct:.1f}%")

        return mask

    def save_mask(self, mask: np.ndarray, path: str | Path) -> None:
        """Save the mask as a .npy file."""
        np.save(path, mask)
        print(f"Mask saved → {path}")

    def load_mask(self, path: str | Path) -> np.ndarray:
        """Load a previously saved mask."""
        return np.load(path)
