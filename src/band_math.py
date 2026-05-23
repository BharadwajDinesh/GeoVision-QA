"""
band_math.py
------------
Spectral index functions that operate on numpy arrays loaded from GeoTIFF.
All inputs are float arrays in the range [0, 1] (already divided by 10000).

Indices implemented:
  - NDVI  : Normalised Difference Vegetation Index
  - NDWI  : Normalised Difference Water Index (Gao)
  - NDBI  : Normalised Difference Built-up Index
  - EVI   : Enhanced Vegetation Index
  - SAVI  : Soil-Adjusted Vegetation Index
  - False colour composite helper
"""

import numpy as np


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Divide a by b, returning 0 wherever b == 0 to avoid NaN/inf.
    All spectral indices use (A - B) / (A + B) — this handles the edge case
    where both bands are zero (e.g. masked pixels).
    """
    return np.where(b != 0, a / b, 0.0)


def normalised_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """
    Generic normalised difference: (A - B) / (A + B)
    Result is always in [-1, 1].
    """
    return _safe_divide(band_a - band_b, band_a + band_b)


# ── Vegetation ────────────────────────────────────────────────────────────────

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)

    Interpretation:
      < 0.0  : Water, snow, bare rock
      0.0–0.1: Bare soil, sand, urban
      0.1–0.3: Sparse / stressed vegetation
      0.3–0.6: Moderate vegetation (crops, grassland)
      > 0.6  : Dense healthy vegetation (forest)

    Bands (Sentinel-2): NIR = B8, Red = B4
    """
    return normalised_difference(nir, red)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
        G: float = 2.5, C1: float = 6.0,
        C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Reduces atmospheric and soil background noise compared to NDVI.
    More sensitive in high-biomass regions where NDVI saturates.

    Bands (Sentinel-2): NIR = B8, Red = B4, Blue = B2
    """
    denom = nir + C1 * red - C2 * blue + L
    return np.where(denom != 0, G * (nir - red) / denom, 0.0)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    L = soil brightness correction factor (0.5 is standard for moderate cover).
    Use L=1 for very sparse cover, L=0.25 for dense cover.

    Bands (Sentinel-2): NIR = B8, Red = B4
    """
    denom = nir + red + L
    return np.where(denom != 0, ((nir - red) / denom) * (1 + L), 0.0)


# ── Water ─────────────────────────────────────────────────────────────────────

def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDWI (McFeeters 1996) = (Green - NIR) / (Green + NIR)

    Positive values (~>0.3) indicate open water surfaces.
    Vegetation and soil give negative values.

    Bands (Sentinel-2): Green = B3, NIR = B8
    """
    return normalised_difference(green, nir)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    MNDWI (Modified NDWI, Xu 2006) = (Green - SWIR1) / (Green + SWIR1)

    Suppresses built-up and vegetation noise better than NDWI.
    Preferred for urban water body detection.

    Bands (Sentinel-2): Green = B3, SWIR1 = B11
    """
    return normalised_difference(green, swir1)


# ── Built-up / Urban ─────────────────────────────────────────────────────────

def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

    Positive values highlight built-up surfaces (concrete, asphalt, rooftops).
    Vegetation gives negative values (inverse of NDVI by design).

    Bands (Sentinel-2): SWIR1 = B11, NIR = B8
    """
    return normalised_difference(swir1, nir)


def urban_index(ndbi_arr: np.ndarray, ndvi_arr: np.ndarray) -> np.ndarray:
    """
    Simple urban mask: pixels where NDBI > 0 AND NDVI < 0.2.
    Returns a binary mask (1 = likely urban, 0 = not urban).
    """
    return ((ndbi_arr > 0) & (ndvi_arr < 0.2)).astype(np.float32)


# ── Composite helpers ─────────────────────────────────────────────────────────

def true_colour(red: np.ndarray, green: np.ndarray, blue: np.ndarray,
                gamma: float = 2.2,
                percentile_clip: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """
    Stack RGB into an (H, W, 3) uint8 array suitable for imshow/PIL.

    Applies percentile stretch (clips outliers) then gamma correction
    to produce a visually pleasing natural colour image.

    Args:
        red, green, blue: Single-band float arrays [0, 1]
        gamma:            Gamma exponent (2.2 = standard monitor gamma)
        percentile_clip:  Lower and upper percentile for contrast stretch

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8
    """
    rgb = np.stack([red, green, blue], axis=-1)
    lo, hi = np.percentile(rgb, [percentile_clip[0], percentile_clip[1]])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.power(rgb, 1.0 / gamma)
    return (rgb * 255).astype(np.uint8)


def false_colour_vegetation(nir: np.ndarray, red: np.ndarray,
                             green: np.ndarray) -> np.ndarray:
    """
    NIR → R, Red → G, Green → B  (classic vegetation false colour).
    Healthy vegetation appears bright red; water is dark blue/black;
    urban is cyan/grey.

    Args:
        nir, red, green: Single-band float arrays [0, 1]

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8
    """
    return true_colour(nir, red, green)


def compute_all_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Convenience wrapper: compute all indices from a band dictionary.

    Args:
        bands: dict mapping band name → float numpy array
               Expected keys: B2 (blue), B3 (green), B4 (red),
                              B8 (NIR), B11 (SWIR1)

    Returns:
        dict mapping index name → numpy array
    """
    b2  = bands["B2"]
    b3  = bands["B3"]
    b4  = bands["B4"]
    b8  = bands["B8"]
    b11 = bands["B11"]

    return {
        "NDVI":  ndvi(b8, b4),
        "EVI":   evi(b8, b4, b2),
        "SAVI":  savi(b8, b4),
        "NDWI":  ndwi(b3, b8),
        "MNDWI": mndwi(b3, b11),
        "NDBI":  ndbi(b11, b8),
        "urban_mask": urban_index(ndbi(b11, b8), ndvi(b8, b4)),
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic pixel values for sanity check
    nir_test  = np.array([0.4, 0.1, 0.05])
    red_test  = np.array([0.1, 0.08, 0.06])
    green_test = np.array([0.08, 0.05, 0.04])
    blue_test  = np.array([0.05, 0.04, 0.03])
    swir1_test = np.array([0.05, 0.12, 0.25])

    ndvi_vals = ndvi(nir_test, red_test)
    print("NDVI test:", ndvi_vals)
    # Expected: [0.60, 0.11, -0.09]  → vegetation, soil, maybe water

    evi_vals  = evi(nir_test, red_test, blue_test)
    print("EVI  test:", evi_vals.round(3))

    ndwi_vals = ndwi(green_test, nir_test)
    print("NDWI test:", ndwi_vals.round(3))
