"""
tile_slicer.py
--------------
Slices a large GeoTIFF into fixed-size patches (tiles) for model training.
Reads directly from GCS and writes tiles back to GCS as .npy files.

Each tile is saved as a .npy file of shape (C, TILE_SIZE, TILE_SIZE)
where C = number of bands in the source image.

Usage:
    from tile_slicer import TileSlicer
    slicer = TileSlicer()
    slicer.slice(full_mask)
"""

import io
import numpy as np
import rasterio
from google.cloud import storage
from pathlib import Path


# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET       = "geovision-data"
GCS_TIF_PATH     = "geovision/phase1/bengaluru_s2_composite_2024.tif"
GCS_TILES_PREFIX = "geovision/phase2/tiles/"
GCS_MASKS_PREFIX = "geovision/phase2/masks/"


class TileSlicer:
    """
    Slices a GeoTIFF (read from GCS) into fixed-size overlapping tiles
    and saves them back to GCS as .npy files.

    Args:
        tile_size:     Width and height of each tile in pixels (default 512)
        overlap:       Overlap between adjacent tiles in pixels (default 64)
        scale:         Divide raw pixel values by this to get reflectance [0,1]
                       Set to 1.0 if the file is already in float [0,1] range
        min_label_pct: Skip tiles where labelled pixels < this fraction
    """

    def __init__(
        self,
        tile_size:     int   = 512,
        overlap:       int   = 64,
        scale:         float = 1.0,
        min_label_pct: float = 0.05,
    ):
        self.tile_size     = tile_size
        self.overlap       = overlap
        self.scale         = scale
        self.min_label_pct = min_label_pct
        self.stride        = tile_size - overlap

        # GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(GCS_BUCKET)

        # Read image metadata from GCS once at init
        print(f"Reading metadata from gs://{GCS_BUCKET}/{GCS_TIF_PATH} ...")
        tif_bytes = self._download_bytes(GCS_TIF_PATH)
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            self.height    = src.height
            self.width     = src.width
            self.n_bands   = src.count
            self.transform = src.transform
            self.crs       = src.crs
            self.bounds    = src.bounds

        # Calculate tile grid dimensions
        self.n_cols = (self.width  - overlap) // self.stride
        self.n_rows = (self.height - overlap) // self.stride

        print(f"TileSlicer initialised")
        print(f"  Image     : {self.width} x {self.height} px, {self.n_bands} bands")
        print(f"  Tile size : {tile_size} px  |  Overlap: {overlap} px  |  Stride: {self.stride} px")
        print(f"  Grid      : {self.n_cols} cols x {self.n_rows} rows = {self.n_cols * self.n_rows} tiles")

    # ── GCS helpers ───────────────────────────────────────────────────────────

    def _download_bytes(self, gcs_path: str) -> bytes:
        """Download a GCS object and return raw bytes."""
        blob = self.bucket.blob(gcs_path)
        return blob.download_as_bytes()

    def _upload_npy(self, array: np.ndarray, gcs_path: str) -> None:
        """Serialize a numpy array and upload to GCS."""
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")

    def _download_npy(self, gcs_path: str) -> np.ndarray:
        """Download a .npy file from GCS and return numpy array."""
        data = self._download_bytes(gcs_path)
        return np.load(io.BytesIO(data))

    # ── Core methods ──────────────────────────────────────────────────────────

    def _read_image(self) -> np.ndarray:
        """
        Read the full GeoTIFF from GCS into memory as a float32 array.
        Shape: (C, H, W). Values scaled to [0, 1] and clipped.
        """
        print("Reading full GeoTIFF from GCS into memory...")
        tif_bytes = self._download_bytes(GCS_TIF_PATH)
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            data = src.read().astype(np.float32)

        if self.scale != 1.0:
            data = data / self.scale

        data = np.clip(data, 0, 1)
        print(f"  Image loaded: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
        return data

    def get_tile_bounds(self, row: int, col: int) -> tuple:
        """
        Get pixel coordinates of a tile given its grid position.
        Returns (y0, x0, y1, x1).
        """
        y0 = row * self.stride
        x0 = col * self.stride
        y1 = y0 + self.tile_size
        x1 = x0 + self.tile_size
        return y0, x0, y1, x1

    def get_tile_geo_bounds(self, row: int, col: int) -> tuple:
        """
        Get geographic bounds (lon/lat) of a tile.
        Returns (west, south, east, north) in the image CRS.
        """
        y0, x0, y1, x1 = self.get_tile_bounds(row, col)
        west,  north   = self.transform * (x0, y0)
        east,  south   = self.transform * (x1, y1)
        return west, south, east, north

    def slice(self, full_mask: np.ndarray, verbose: bool = True) -> dict:
        """
        Slice the GeoTIFF and segmentation mask into tiles.
        Reads image from GCS, writes tiles + masks back to GCS.

        Args:
            full_mask: Full-scene segmentation mask (H, W) numpy array
                       with integer class labels 0-4
            verbose:   Print progress every 50 tiles

        Returns:
            stats dict with keys: saved, skipped_edge, skipped_empty, total
        """
        img_data = self._read_image()   # (C, H, W)

        stats = {'saved': 0, 'skipped_edge': 0, 'skipped_empty': 0}

        print(f"\nSlicing into tiles and uploading to GCS...")
        print(f"  Tiles prefix : gs://{GCS_BUCKET}/{GCS_TILES_PREFIX}")
        print(f"  Masks prefix : gs://{GCS_BUCKET}/{GCS_MASKS_PREFIX}")

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                y0, x0, y1, x1 = self.get_tile_bounds(row, col)

                # Skip edge tiles that don't fill the full tile size
                if y1 > self.height or x1 > self.width:
                    stats['skipped_edge'] += 1
                    continue

                img_tile  = img_data[:, y0:y1, x0:x1]   # (C, 512, 512)
                mask_tile = full_mask[y0:y1, x0:x1]      # (512, 512)

                # Skip tiles with too little label coverage
                label_pct = (mask_tile > 0).sum() / mask_tile.size
                if label_pct < self.min_label_pct:
                    stats['skipped_empty'] += 1
                    continue

                tile_id    = f"tile_{row:03d}_{col:03d}"
                tile_path  = f"{GCS_TILES_PREFIX}{tile_id}.npy"
                mask_path  = f"{GCS_MASKS_PREFIX}{tile_id}.npy"

                self._upload_npy(img_tile,  tile_path)
                self._upload_npy(mask_tile, mask_path)
                stats['saved'] += 1

                if verbose and stats['saved'] % 50 == 0:
                    print(f"  Saved {stats['saved']} tiles to GCS...")

        stats['total'] = self.n_rows * self.n_cols
        print(f"\nSlicing complete:")
        print(f"  Saved        : {stats['saved']} tiles")
        print(f"  Skipped edge : {stats['skipped_edge']}")
        print(f"  Skipped empty: {stats['skipped_empty']}")
        return stats

    def slice_single(self, row: int, col: int, full_mask: np.ndarray | None = None) -> tuple:
        """
        Extract a single tile by grid position. Useful for debugging.

        Args:
            row, col:  Grid position
            full_mask: Optional mask array

        Returns:
            (img_tile, mask_tile) or just img_tile if no mask provided
        """
        tif_bytes = self._download_bytes(GCS_TIF_PATH)
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            img_data = src.read().astype(np.float32)
        if self.scale != 1.0:
            img_data = np.clip(img_data / self.scale, 0, 1)

        y0, x0, y1, x1 = self.get_tile_bounds(row, col)
        img_tile = img_data[:, y0:y1, x0:x1]

        if full_mask is not None:
            return img_tile, full_mask[y0:y1, x0:x1]
        return img_tile

    def list_saved_tiles(self) -> list:
        """List all tile keys currently saved in GCS."""
        blobs = self.client.list_blobs(GCS_BUCKET, prefix=GCS_TILES_PREFIX)
        keys  = [b.name for b in blobs if b.name.endswith(".npy")]
        print(f"Found {len(keys)} tiles in gs://{GCS_BUCKET}/{GCS_TILES_PREFIX}")
        return keys
