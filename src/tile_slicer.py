"""
tile_slicer.py
--------------
Slices a large GeoTIFF into fixed-size patches (tiles) for model training.

Each tile is saved as a .npy file of shape (C, TILE_SIZE, TILE_SIZE)
where C = number of bands in the source image.

Usage:
    from tile_slicer import TileSlicer
    slicer = TileSlicer(tif_path, tile_size=512, overlap=64)
    slicer.slice(tiles_dir, masks_dir, full_mask)
"""

import numpy as np
import rasterio
from pathlib import Path


class TileSlicer:
    """
    Slices a GeoTIFF into fixed-size overlapping tiles.

    Args:
        tif_path:  Path to the source GeoTIFF file
        tile_size: Width and height of each tile in pixels (default 512)
        overlap:   Overlap between adjacent tiles in pixels (default 64)
                   Prevents objects at tile edges from being cut off
        scale:     Divide raw pixel values by this to get reflectance [0,1]
                   Set to 1 if the file is already in float [0,1] range
        min_label_pct: Skip tiles where labelled pixels are below this
                       fraction (avoids saving uninformative background tiles)
    """

    def __init__(
        self,
        tif_path: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
        scale: float = 1.0,
        min_label_pct: float = 0.05,
    ):
        self.tif_path      = Path(tif_path)
        self.tile_size     = tile_size
        self.overlap       = overlap
        self.scale         = scale
        self.min_label_pct = min_label_pct
        self.stride        = tile_size - overlap

        # Read image metadata once
        with rasterio.open(self.tif_path) as src:
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
        print(f"  Image     : {self.width} × {self.height} px, {self.n_bands} bands")
        print(f"  Tile size : {tile_size} px  |  Overlap: {overlap} px  |  Stride: {self.stride} px")
        print(f"  Grid      : {self.n_cols} cols × {self.n_rows} rows = {self.n_cols * self.n_rows} tiles")

    def _read_image(self) -> np.ndarray:
        """
        Read the full GeoTIFF into memory as a float32 array.
        Shape: (C, H, W) where C = number of bands.
        Values are scaled to [0, 1] and clipped.
        """
        with rasterio.open(self.tif_path) as src:
            data = src.read().astype(np.float32)

        if self.scale != 1.0:
            data = data / self.scale

        return np.clip(data, 0, 1)

    def get_tile_bounds(self, row: int, col: int) -> tuple:
        """
        Get the pixel coordinates of a tile given its grid position.

        Returns:
            (y0, x0, y1, x1) pixel coordinates
        """
        y0 = row * self.stride
        x0 = col * self.stride
        y1 = y0 + self.tile_size
        x1 = x0 + self.tile_size
        return y0, x0, y1, x1

    def get_tile_geo_bounds(self, row: int, col: int) -> tuple:
        """
        Get the geographic bounds (lon/lat) of a tile.
        Useful for georeferencing individual tiles later.

        Returns:
            (west, south, east, north) in the image CRS
        """
        y0, x0, y1, x1 = self.get_tile_bounds(row, col)
        west,  north   = self.transform * (x0, y0)
        east,  south   = self.transform * (x1, y1)
        return west, south, east, north

    def slice(
        self,
        tiles_dir: str | Path,
        masks_dir: str | Path,
        full_mask: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Slice the GeoTIFF and a corresponding segmentation mask into tiles.

        Tiles that are smaller than tile_size (edge tiles) are skipped.
        Tiles where labelled pixels < min_label_pct are skipped.

        Args:
            tiles_dir:  Directory to save image tile .npy files
            masks_dir:  Directory to save mask tile .npy files
            full_mask:  Full-scene segmentation mask (H, W) numpy array
                        with integer class labels
            verbose:    Print progress every 50 tiles

        Returns:
            stats dict with keys: saved, skipped_edge, skipped_empty, total
        """
        tiles_dir = Path(tiles_dir)
        masks_dir = Path(masks_dir)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        img_data = self._read_image()   # (C, H, W)

        stats = {'saved': 0, 'skipped_edge': 0, 'skipped_empty': 0}

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                y0, x0, y1, x1 = self.get_tile_bounds(row, col)

                # Skip edge tiles that don't fill the full tile size
                if y1 > self.height or x1 > self.width:
                    stats['skipped_edge'] += 1
                    continue

                img_tile  = img_data[:, y0:y1, x0:x1]    # (C, 512, 512)
                mask_tile = full_mask[y0:y1, x0:x1]       # (512, 512)

                # Skip tiles with too little label coverage
                label_pct = (mask_tile > 0).sum() / mask_tile.size
                if label_pct < self.min_label_pct:
                    stats['skipped_empty'] += 1
                    continue

                tile_id = f"tile_{row:03d}_{col:03d}"
                np.save(tiles_dir / f"{tile_id}.npy", img_tile)
                np.save(masks_dir / f"{tile_id}.npy", mask_tile)
                stats['saved'] += 1

                if verbose and stats['saved'] % 50 == 0:
                    print(f"  Saved {stats['saved']} tiles...")

        stats['total'] = self.n_rows * self.n_cols
        print(f"\nSlicing complete:")
        print(f"  Saved        : {stats['saved']} tiles")
        print(f"  Skipped edge : {stats['skipped_edge']}")
        print(f"  Skipped empty: {stats['skipped_empty']}")
        return stats

    def slice_single(
        self,
        row: int,
        col: int,
        full_mask: np.ndarray | None = None,
    ) -> tuple:
        """
        Extract a single tile by grid position. Useful for debugging.

        Args:
            row, col:  Grid position
            full_mask: Optional mask array; if None only image tile returned

        Returns:
            (img_tile, mask_tile) or just img_tile if no mask provided
            img_tile shape: (C, tile_size, tile_size)
            mask_tile shape: (tile_size, tile_size)
        """
        with rasterio.open(self.tif_path) as src:
            img_data = src.read().astype(np.float32)
        if self.scale != 1.0:
            img_data = np.clip(img_data / self.scale, 0, 1)

        y0, x0, y1, x1 = self.get_tile_bounds(row, col)
        img_tile = img_data[:, y0:y1, x0:x1]

        if full_mask is not None:
            return img_tile, full_mask[y0:y1, x0:x1]
        return img_tile
