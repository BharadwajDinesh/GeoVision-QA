"""
visualize.py
------------
Plotting utilities for satellite imagery and spectral indices.
All functions return (fig, ax) so callers can save or further annotate.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional


# ── Colour maps ───────────────────────────────────────────────────────────────

# NDVI: brown (bare) → yellow (sparse) → green (dense)
NDVI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ndvi",
    ["#8B4513", "#F5DEB3", "#ADFF2F", "#228B22"],
    N=256,
)

# Water index: dry (orange) → neutral (white) → water (blue)
WATER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "water",
    ["#FF8C00", "#FFFACD", "#4169E1"],
    N=256,
)

# Urban index: low (white) → high (dark red)
URBAN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "urban",
    ["#F5F5F5", "#FF4500", "#8B0000"],
    N=256,
)

INDEX_CMAPS = {
    "NDVI":       (NDVI_CMAP,  -1, 1),
    "EVI":        (NDVI_CMAP,  -1, 1),
    "SAVI":       (NDVI_CMAP,  -1, 1),
    "NDWI":       (WATER_CMAP, -1, 1),
    "MNDWI":      (WATER_CMAP, -1, 1),
    "NDBI":       (URBAN_CMAP, -1, 1),
    "urban_mask": ("Reds",      0, 1),
}


# ── Single-band plot ──────────────────────────────────────────────────────────

def plot_index(
    index_array: np.ndarray,
    index_name: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Plot a single spectral index with an appropriate colour map and colour bar.

    Args:
        index_array: 2D numpy array of index values
        index_name:  One of the keys in INDEX_CMAPS (e.g. "NDVI")
        title:       Optional plot title (defaults to index_name)
        save_path:   If provided, saves the figure to this path
        figsize:     Matplotlib figure size

    Returns:
        (fig, ax) tuple
    """
    cmap, vmin, vmax = INDEX_CMAPS.get(index_name, ("viridis", None, None))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label=index_name)
    ax.set_title(title or index_name, fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig, ax


# ── RGB composite ─────────────────────────────────────────────────────────────

def plot_rgb(
    rgb_array: np.ndarray,
    title: str = "True colour composite",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 8),
) -> tuple:
    """
    Plot an (H, W, 3) uint8 RGB array.

    Args:
        rgb_array:  Output of band_math.true_colour() or false_colour_vegetation()
        title:      Plot title
        save_path:  Optional save path
        figsize:    Figure size

    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_array)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig, ax


# ── Index dashboard ───────────────────────────────────────────────────────────

def plot_index_dashboard(
    rgb: np.ndarray,
    indices: dict[str, np.ndarray],
    region_name: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (18, 10),
) -> tuple:
    """
    Multi-panel figure: true colour + up to 6 spectral indices.

    Args:
        rgb:         (H, W, 3) uint8 RGB array
        indices:     dict of index_name → 2D array (from band_math.compute_all_indices)
        region_name: Used in the suptitle
        save_path:   Optional path to save the figure
        figsize:     Overall figure size

    Returns:
        (fig, axes) tuple
    """
    n_indices = len(indices)
    n_cols = 4
    n_rows = (n_indices + 1 + n_cols - 1) // n_cols  # +1 for the RGB panel

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    # Panel 0: true colour
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("True colour (RGB)", fontsize=10)
    ax_rgb.axis("off")

    # Remaining panels: one per index
    axes = [ax_rgb]
    for i, (name, arr) in enumerate(indices.items(), start=1):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        cmap, vmin, vmax = INDEX_CMAPS.get(name, ("viridis", None, None))
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        axes.append(ax)

    # Hide any leftover empty panels
    total_panels = n_rows * n_cols
    for j in range(n_indices + 1, total_panels):
        row, col = divmod(j, n_cols)
        fig.add_subplot(gs[row, col]).axis("off")

    label = f" — {region_name}" if region_name else ""
    fig.suptitle(f"Sentinel-2 spectral indices{label}", fontsize=14, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dashboard → {save_path}")

    return fig, axes


# ── Histogram ─────────────────────────────────────────────────────────────────

def plot_band_histograms(
    bands: dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
) -> tuple:
    """
    Plot value distributions for each band. Useful for checking that
    cloud masking and scaling worked correctly.
    Values should be centred between 0 and 0.3 for typical land scenes.

    Args:
        bands:     dict of band_name → 2D float array
        save_path: Optional save path
        figsize:   Figure size

    Returns:
        (fig, axes) tuple
    """
    names = list(bands.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]

    colours = plt.cm.tab10(np.linspace(0, 1, n))

    for ax, name, colour in zip(axes, names, colours):
        data = bands[name].ravel()
        data = data[~np.isnan(data)]               # drop masked pixels
        data = data[(data >= 0) & (data <= 1)]     # clip to valid range
        ax.hist(data, bins=80, color=colour, alpha=0.75, edgecolor="none")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Reflectance", fontsize=8)
        ax.set_xlim(0, 0.5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Band reflectance distributions", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histograms → {save_path}")

    return fig, axes
