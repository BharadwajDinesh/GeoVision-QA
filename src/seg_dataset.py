"""
seg_dataset.py
--------------
PyTorch Dataset class that loads image tiles and segmentation masks
directly from GCS for SegFormer fine-tuning.

Handles:
  - Loading .npy tile/mask pairs from GCS
  - Band selection (RGB or all 6 Sentinel-2 bands)
  - Image normalisation
  - Train/val split
  - Optional augmentation

Usage:
    from seg_dataset import SatelliteSegDataset, get_dataloaders
    train_loader, val_loader = get_dataloaders()
"""

import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from google.cloud import storage


# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET       = "geovision-data"
GCS_TILES_PREFIX = "geovision/phase2/tiles/"
GCS_MASKS_PREFIX = "geovision/phase2/masks/"


# ── Normalisation stats ───────────────────────────────────────────────────────
# SegFormer pretrained on ImageNet — use ImageNet stats for RGB bands
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]

# Approximate stats for all 6 Sentinel-2 bands (B2,B3,B4,B8,B11,B12)
S2_MEAN  = [0.085, 0.095, 0.090, 0.250, 0.200, 0.150]
S2_STD   = [0.040, 0.045, 0.050, 0.080, 0.075, 0.070]


# ── Class config ──────────────────────────────────────────────────────────────
NUM_CLASSES  = 5
CLASS_NAMES  = ['background', 'building', 'road', 'vegetation', 'water']

# Down-weight background (most common), up-weight rare classes like water
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 2.0, 1.5, 3.0], dtype=torch.float32)


class SatelliteSegDataset(Dataset):
    """
    PyTorch Dataset for satellite image segmentation.
    Loads paired .npy tile/mask files directly from GCS.

    Tile files : (C, 512, 512) float32 arrays
    Mask files : (512, 512)    uint8 arrays with class labels 0-4

    Args:
        use_rgb: If True, use only RGB bands (B4, B3, B2 = 3 channels)
                 If False, use all 6 Sentinel-2 bands
        augment: If True, apply random flips and 90° rotations
    """

    def __init__(self, use_rgb: bool = False, augment: bool = False):
        self.use_rgb = use_rgb
        self.augment = augment

        # GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(GCS_BUCKET)

        # List all tile keys from GCS
        all_tile_keys = [
            b.name
            for b in self.client.list_blobs(GCS_BUCKET, prefix=GCS_TILES_PREFIX)
            if b.name.endswith(".npy")
        ]

        # Only keep tiles that have a corresponding mask in GCS
        mask_keys = set(
            b.name
            for b in self.client.list_blobs(GCS_BUCKET, prefix=GCS_MASKS_PREFIX)
            if b.name.endswith(".npy")
        )

        self.tile_keys = [
            t for t in all_tile_keys
            if t.replace(GCS_TILES_PREFIX, GCS_MASKS_PREFIX) in mask_keys
        ]

        if len(self.tile_keys) == 0:
            raise ValueError(
                f"No matching tile/mask pairs found in GCS bucket '{GCS_BUCKET}'.\n"
                f"  Tiles prefix: {GCS_TILES_PREFIX}\n"
                f"  Masks prefix: {GCS_MASKS_PREFIX}\n"
                f"Run TileSlicer.slice() first to generate tiles."
            )

        # Normalisation stats
        if use_rgb:
            self.mean       = np.array(RGB_MEAN, dtype=np.float32)
            self.std        = np.array(RGB_STD,  dtype=np.float32)
            self.n_channels = 3
        else:
            self.mean       = np.array(S2_MEAN, dtype=np.float32)
            self.std        = np.array(S2_STD,  dtype=np.float32)
            self.n_channels = 6

        print(f"Dataset loaded : {len(self.tile_keys)} tile-mask pairs from GCS")
        print(f"Channels       : {self.n_channels} ({'RGB' if use_rgb else 'all 6 S2 bands'})")
        print(f"Augmentation   : {augment}")

    # ── GCS helpers ───────────────────────────────────────────────────────────

    def _load_npy(self, gcs_key: str) -> np.ndarray:
        """Download a .npy file from GCS and return as numpy array."""
        blob = self.bucket.blob(gcs_key)
        data = blob.download_as_bytes()
        return np.load(io.BytesIO(data))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.tile_keys)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single tile/mask pair from GCS.

        Returns:
            dict with keys:
              'pixel_values': FloatTensor (C, 512, 512)
              'labels':       LongTensor  (512, 512)
              'tile_id':      str identifier
        """
        tile_key = self.tile_keys[idx]
        mask_key = tile_key.replace(GCS_TILES_PREFIX, GCS_MASKS_PREFIX)
        tile_id  = tile_key.split("/")[-1].replace(".npy", "")

        img = self._load_npy(tile_key).astype(np.float32)   # (6, 512, 512)
        msk = self._load_npy(mask_key)                       # (512, 512) uint8

        # Band selection: RGB uses B4(red)=idx2, B3(green)=idx1, B2(blue)=idx0
        if self.use_rgb:
            img = img[[2, 1, 0], :, :]   # (3, 512, 512)

        # Normalise: (value - mean) / std
        img = (img - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-9)

        # Augmentation: random flips + 90° rotations
        if self.augment:
            if np.random.rand() > 0.5:                          # horizontal flip
                img = np.flip(img, axis=2).copy()
                msk = np.flip(msk, axis=1).copy()
            if np.random.rand() > 0.5:                          # vertical flip
                img = np.flip(img, axis=1).copy()
                msk = np.flip(msk, axis=0).copy()
            k = np.random.choice([0, 1, 2, 3])                  # 90° rotation
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                msk = np.rot90(msk, k, axes=(0, 1)).copy()

        return {
            'pixel_values': torch.from_numpy(img),
            'labels':       torch.from_numpy(msk.astype(np.int64)),
            'tile_id':      tile_id,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Return class weights tensor for weighted cross-entropy loss."""
        return CLASS_WEIGHTS

    def summary(self, n_sample: int = 20) -> None:
        """Print dataset statistics by sampling tiles from GCS."""
        print(f"\nDataset summary:")
        print(f"  Total tiles : {len(self.tile_keys)}")
        print(f"  Channels    : {self.n_channels}")
        print(f"  Tile size   : 512 x 512 px")
        print(f"  Classes     : {CLASS_NAMES}")

        counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
        n_sample = min(n_sample, len(self.tile_keys))
        for i in range(n_sample):
            mask_key = self.tile_keys[i].replace(GCS_TILES_PREFIX, GCS_MASKS_PREFIX)
            msk      = self._load_npy(mask_key)
            for c in range(NUM_CLASSES):
                counts[c] += (msk == c).sum()

        total = counts.sum()
        print(f"\n  Class distribution (first {n_sample} tiles):")
        for name, count in zip(CLASS_NAMES, counts):
            print(f"    {name:12s}: {count / total * 100:.1f}%")


def get_dataloaders(
    val_split:   float = 0.2,
    batch_size:  int   = 8,
    num_workers: int   = 2,
    use_rgb:     bool  = False,
    seed:        int   = 42,
) -> tuple:
    """
    Create train and validation DataLoaders backed by GCS.

    Args:
        val_split:   Fraction of data for validation (default 0.2)
        batch_size:  Samples per batch (default 8, reduce if OOM)
        num_workers: DataLoader worker processes (default 2)
        use_rgb:     If True use RGB only; else all 6 Sentinel-2 bands
        seed:        Random seed for reproducible splits

    Returns:
        (train_loader, val_loader)
    """
    # Full dataset (no augmentation) for splitting
    full_dataset = SatelliteSegDataset(use_rgb=use_rgb, augment=False)

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    # Wrap train indices with augmentation enabled
    train_dataset_aug = SatelliteSegDataset(use_rgb=use_rgb, augment=True)
    train_dataset     = Subset(train_dataset_aug, train_subset.indices)

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
