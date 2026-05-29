"""
seg_dataset.py
--------------
PyTorch Dataset class that loads image tiles and segmentation masks
for SegFormer fine-tuning.

Handles:
  - Loading .npy tile/mask pairs
  - Band selection (use RGB or all 6 bands)
  - Image normalisation
  - Train/val split
  - Optional augmentation

Usage:
    from seg_dataset import SatelliteSegDataset, get_dataloaders
    train_loader, val_loader = get_dataloaders(tiles_dir, masks_dir)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional


# ── ImageNet-style normalisation stats ───────────────────────────────────────
# SegFormer was pretrained on ImageNet RGB images normalised with these values.
# We use the same stats for the RGB bands (B4, B3, B2 → R, G, B).
# For all 6 bands we use the mean/std computed from our Bengaluru tile.

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

# Approximate stats for all 6 Sentinel-2 bands (B2,B3,B4,B8,B11,B12)
S2_MEAN  = [0.085, 0.095, 0.090, 0.250, 0.200, 0.150]
S2_STD   = [0.040, 0.045, 0.050, 0.080, 0.075, 0.070]

NUM_CLASSES = 5   # background, building, road, vegetation, water

CLASS_NAMES = ['background', 'building', 'road', 'vegetation', 'water']

# Class weights for loss function
# Down-weight background (most common), up-weight rare classes like water
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 2.0, 1.5, 3.0], dtype=torch.float32)


class SatelliteSegDataset(Dataset):
    """
    PyTorch Dataset for satellite image segmentation.

    Loads paired .npy files from tiles_dir and masks_dir.
    Tile files: (C, 512, 512) float32 arrays
    Mask files: (512, 512) uint8 arrays with class labels 0-4

    Args:
        tiles_dir:   Directory containing image tile .npy files
        masks_dir:   Directory containing mask .npy files
        use_rgb:     If True, use only RGB bands (3 channels)
                     If False, use all 6 Sentinel-2 bands
        augment:     If True, apply random flips and rotations
    """

    def __init__(
        self,
        tiles_dir: str | Path,
        masks_dir: str | Path,
        use_rgb:   bool = False,
        augment:   bool = False,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir)
        self.use_rgb   = use_rgb
        self.augment   = augment

        # Find all tile files and verify matching masks exist
        self.tile_ids = []
        for tile_file in sorted(self.tiles_dir.glob("*.npy")):
            mask_file = self.masks_dir / tile_file.name
            if mask_file.exists():
                self.tile_ids.append(tile_file.stem)

        if len(self.tile_ids) == 0:
            raise ValueError(
                f"No matching tile/mask pairs found in "
                f"{tiles_dir} and {masks_dir}"
            )

        # Set normalisation stats based on band selection
        if use_rgb:
            self.mean = np.array(RGB_MEAN, dtype=np.float32)
            self.std  = np.array(RGB_STD,  dtype=np.float32)
            self.n_channels = 3
        else:
            self.mean = np.array(S2_MEAN, dtype=np.float32)
            self.std  = np.array(S2_STD,  dtype=np.float32)
            self.n_channels = 6

        print(f"Dataset loaded: {len(self.tile_ids)} tiles")
        print(f"Channels      : {self.n_channels} ({'RGB' if use_rgb else 'all S2 bands'})")
        print(f"Augmentation  : {augment}")

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single tile/mask pair.

        Returns:
            dict with keys:
              'pixel_values': torch.FloatTensor of shape (C, 512, 512)
              'labels':       torch.LongTensor  of shape (512, 512)
              'tile_id':      str identifier
        """
        tile_id   = self.tile_ids[idx]
        img_array = np.load(self.tiles_dir / f"{tile_id}.npy")  # (6, 512, 512)
        msk_array = np.load(self.masks_dir / f"{tile_id}.npy")  # (512, 512)

        # Band selection
        if self.use_rgb:
            # Sentinel-2 band order: B2(0), B3(1), B4(2), B8(3), B11(4), B12(5)
            # RGB = B4(red), B3(green), B2(blue) = indices 2, 1, 0
            img_array = img_array[[2, 1, 0], :, :]   # (3, 512, 512)

        # Normalise: (value - mean) / std
        # mean/std shape: (C,) → expand to (C, 1, 1) for broadcasting
        img_array = (img_array - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-9)
        img_array = img_array.astype(np.float32)

        # Augmentation: random horizontal/vertical flip + 90° rotation
        if self.augment:
            if np.random.rand() > 0.5:
                img_array = np.flip(img_array, axis=2).copy()   # horizontal flip
                msk_array = np.flip(msk_array, axis=1).copy()
            if np.random.rand() > 0.5:
                img_array = np.flip(img_array, axis=1).copy()   # vertical flip
                msk_array = np.flip(msk_array, axis=0).copy()
            k = np.random.choice([0, 1, 2, 3])                  # 90° rotations
            if k > 0:
                img_array = np.rot90(img_array, k, axes=(1, 2)).copy()
                msk_array = np.rot90(msk_array, k, axes=(0, 1)).copy()

        return {
            'pixel_values': torch.from_numpy(img_array),
            'labels':       torch.from_numpy(msk_array.astype(np.int64)),
            'tile_id':      tile_id,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Return class weights tensor for weighted cross-entropy loss."""
        return CLASS_WEIGHTS

    def summary(self) -> None:
        """Print dataset statistics."""
        print(f"\nDataset summary:")
        print(f"  Total tiles : {len(self.tile_ids)}")
        print(f"  Channels    : {self.n_channels}")
        print(f"  Tile size   : 512 × 512 px")
        print(f"  Classes     : {CLASS_NAMES}")

        # Sample class distribution across first 20 tiles
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        n_sample = min(20, len(self.tile_ids))
        for i in range(n_sample):
            msk = np.load(self.masks_dir / f"{self.tile_ids[i]}.npy")
            for c in range(NUM_CLASSES):
                counts[c] += (msk == c).sum()
        total = counts.sum()
        print(f"\n  Class distribution (first {n_sample} tiles):")
        for name, count in zip(CLASS_NAMES, counts):
            print(f"    {name:12s}: {count/total*100:.1f}%")


def get_dataloaders(
    tiles_dir:   str | Path,
    masks_dir:   str | Path,
    val_split:   float = 0.2,
    batch_size:  int   = 8,
    num_workers: int   = 2,
    use_rgb:     bool  = False,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with a random split.

    Args:
        tiles_dir:   Directory with image tile .npy files
        masks_dir:   Directory with mask .npy files
        val_split:   Fraction of data to use for validation (default 0.2)
        batch_size:  Samples per batch (default 8, reduce if OOM)
        num_workers: DataLoader worker processes (default 2)
        use_rgb:     If True, use RGB only; else all 6 bands
        seed:        Random seed for reproducible splits

    Returns:
        (train_loader, val_loader) tuple of DataLoaders
    """
    # Full dataset without augmentation first (for splitting)
    full_dataset = SatelliteSegDataset(tiles_dir, masks_dir, use_rgb=use_rgb)

    # Split into train/val
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Wrap train split with augmentation
    train_dataset = SatelliteSegDataset(
        tiles_dir, masks_dir, use_rgb=use_rgb, augment=True
    )
    # Use same indices as the random split
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train : {n_train} tiles, {len(train_loader)} batches")
    print(f"  Val   : {n_val} tiles,  {len(val_loader)} batches")
    print(f"  Batch size  : {batch_size}")

    return train_loader, val_loader
