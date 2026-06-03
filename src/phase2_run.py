import sys
sys.path.append('/home/bharathd7900/geovision/src')

from osm_labels import OSMLabeler
from tile_slicer import TileSlicer
from seg_dataset import get_dataloaders

# ── Step 1: Build OSM mask ────────────────────────────────────────────────────
print("="*50)
print("STEP 1: Building OSM segmentation mask")
print("="*50)

labeler = OSMLabeler()


if labeler.mask_exists_in_gcs():
    print("Mask already exists in GCS, loading...")
    full_mask = labeler.load_mask_from_gcs()
else:
    full_mask = labeler.build_mask()
    labeler.save_mask_to_gcs(full_mask)

# ── Step 2: Slice into tiles ──────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: Slicing GeoTIFF into tiles")
print("="*50)
slicer  = TileSlicer(min_label_pct=0.01)
stats  = slicer.slice(full_mask)

# ── Step 3: Verify dataset ────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Verifying dataset")
print("="*50)
train_loader, val_loader = get_dataloaders(batch_size=4)

# Check one batch loads correctly
batch = next(iter(train_loader))
print(f"Batch pixel_values shape : {batch['pixel_values'].shape}")
print(f"Batch labels shape       : {batch['labels'].shape}")
print("Dataset verified — ready for training!")