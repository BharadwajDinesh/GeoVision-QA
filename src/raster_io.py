"""
raster_io.py
------------
Read/write GeoTIFF files using rasterio.
Bridges the gap between Earth Engine exports and numpy arrays
used by band_math.py and visualize.py.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Optional


# ── Band name mapping for exported composites ─────────────────────────────────
# When EE exports a multi-band GeoTIFF the bands are written in the order
# you specified in DEFAULT_BANDS. We re-attach human-readable names here.

DEFAULT_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


def read_geotiff(
    path: Path | str,
    band_names: Optional[list[str]] = None,
    as_float: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Read a multi-band GeoTIFF into a dict of numpy arrays.

    Args:
        path:       Path to the .tif file
        band_names: List of names to assign to bands in order.
                    Defaults to DEFAULT_BAND_ORDER.
        as_float:   If True, cast uint16 data to float32 and
                    values > 1 are divided by 10000 (EE scale factor).
                    If your file is already in [0,1], set False.

    Returns:
        bands: dict mapping band_name → 2D numpy array (H, W)
        meta:  rasterio metadata dict (CRS, transform, dtype, etc.)

    Example:
        bands, meta = read_geotiff("bengaluru_s2_composite_2024.tif")
        ndvi_arr = band_math.ndvi(bands["B8"], bands["B4"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    names = band_names or DEFAULT_BAND_ORDER

    with rasterio.open(path) as src:
        meta = src.meta.copy()
        n_bands = src.count

        if len(names) != n_bands:
            print(
                f"Warning: file has {n_bands} bands but {len(names)} names given. "
                f"Using generic names B1…B{n_bands}."
            )
            names = [f"B{i+1}" for i in range(n_bands)]

        bands = {}
        for i, name in enumerate(names, start=1):
            arr = src.read(i).astype(np.float32)
            if as_float and arr.max() > 1.5:
                # Scale from uint16 [0, 10000] → float [0, 1]
                arr = arr / 10000.0
            # Replace nodata with NaN
            if src.nodata is not None:
                arr[arr == src.nodata / 10000.0] = np.nan
            bands[name] = arr

    print(f"Read {n_bands}-band GeoTIFF: {path.name}  "
          f"({meta['height']} × {meta['width']} px, CRS: {meta['crs']})")
    return bands, meta


def write_geotiff(
    arrays: dict[str, np.ndarray],
    output_path: Path | str,
    reference_meta: dict,
    dtype: str = "float32",
) -> None:
    """
    Write a dict of 2D arrays to a multi-band GeoTIFF.

    Useful for saving computed index stacks (NDVI, NDWI, NDBI…)
    so they can be loaded by Phase 2 model training.

    Args:
        arrays:         dict of band_name → 2D numpy array
        output_path:    Destination .tif path
        reference_meta: Metadata from read_geotiff() to copy CRS/transform
        dtype:          Output data type
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names  = list(arrays.keys())
    sample = next(iter(arrays.values()))
    h, w   = sample.shape

    meta = reference_meta.copy()
    meta.update({
        "count": len(arrays),
        "dtype": dtype,
        "driver": "GTiff",
        "compress": "lzw",
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        for i, (name, arr) in enumerate(arrays.items(), start=1):
            dst.write(arr.astype(dtype), i)
            dst.update_tags(i, name=name)

    print(f"Saved {len(arrays)}-band GeoTIFF → {output_path}  "
          f"(bands: {', '.join(names)})")


def get_tile_info(path: Path | str) -> None:
    """
    Print a human-readable summary of a GeoTIFF file.
    Useful for quick sanity checks after downloading from Drive/GCS.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        bounds = src.bounds
        print(f"\n── {path.name} ──")
        print(f"  Dimensions : {src.width} × {src.height} px")
        print(f"  Bands      : {src.count}")
        print(f"  CRS        : {src.crs}")
        print(f"  Resolution : {src.res[0]:.4f}° / pixel")
        print(f"  Bounds     : W={bounds.left:.4f} E={bounds.right:.4f} "
              f"S={bounds.bottom:.4f} N={bounds.top:.4f}")
        print(f"  dtype      : {src.dtypes[0]}")
        print(f"  nodata     : {src.nodata}")
