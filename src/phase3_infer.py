"""
phase3_infer.py
---------------
Phase 3: Change Detection Inference
Loads 2015 and 2024 Koramangala images from GCS,
runs SegFormer segmentation on both,
compares masks to detect land cover changes,
saves results back to GCS.

Usage:
    python src/phase3_infer.py
"""

import sys
sys.path.append('/home/bharathd7900/geovision/src')

import io
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.warp import reproject, Resampling
from transformers import SegformerForSemanticSegmentation
from google.cloud import storage

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET   = "geovision-data"
GCS_2015     = "geovision/phase3/koramangala_s2_2015.tif"
GCS_2024     = "geovision/phase3/koramangala_s2_2024.tif"
GCS_CKPT     = "geovision/phase2/checkpoints/best_segformer.pt"
GCS_OUT      = "geovision/phase3"

NUM_CLASSES  = 5
CLASS_NAMES  = ['background', 'building', 'road', 'vegetation', 'water']
RGB_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── GCS helpers ───────────────────────────────────────────────────────────────
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)

def load_tif(gcs_path):
    data = bucket.blob(gcs_path).download_as_bytes()
    with rasterio.open(io.BytesIO(data)) as src:
        img       = src.read().astype(np.float32)
        transform = src.transform
        crs       = src.crs
        height    = src.height
        width     = src.width
    return img, transform, crs, height, width

def save_npy(array, gcs_path):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    bucket.blob(gcs_path).upload_from_file(
        buf, content_type='application/octet-stream')
    print(f"  Saved → gs://{GCS_BUCKET}/{gcs_path}")

# ── Step 1: Load images ───────────────────────────────────────────────────────
print("="*50)
print("STEP 1: Loading images from GCS")
print("="*50)

img_2015, tf_2015, crs_2015, H, W = load_tif(GCS_2015)
img_2024, tf_2024, crs_2024, _, _ = load_tif(GCS_2024)

img_2015 = np.nan_to_num(img_2015, nan=0.0)
img_2024 = np.nan_to_num(img_2024, nan=0.0)

print(f"2015: {img_2015.shape}")
print(f"2024: {img_2024.shape}")

# ── Step 2: Align 2024 to 2015 grid ──────────────────────────────────────────
print("\n" + "="*50)
print("STEP 2: Aligning images")
print("="*50)

img_2024_aligned = np.zeros_like(img_2015)
for b in range(img_2024.shape[0]):
    reproject(
        source        = img_2024[b],
        destination   = img_2024_aligned[b],
        src_transform = tf_2024,
        src_crs       = crs_2024,
        dst_transform = tf_2015,
        dst_crs       = crs_2015,
        resampling    = Resampling.bilinear
    )

# Fix no-data pixels
no_data = (img_2024_aligned.sum(axis=0) == 0)
for b in range(img_2024_aligned.shape[0]):
    band_mean = img_2024_aligned[b][~no_data].mean()
    img_2024_aligned[b][no_data] = band_mean

img_2024_aligned = np.clip(img_2024_aligned, 0, 1)
img_2015         = np.clip(img_2015, 0, 1)
print(f"Aligned: {img_2024_aligned.shape}")

# ── Step 3: Load SegFormer ────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 3: Loading SegFormer checkpoint")
print("="*50)

ckpt = bucket.blob(GCS_CKPT).download_as_bytes()
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(
    torch.load(io.BytesIO(ckpt), map_location=DEVICE), strict=False)
model = model.to(DEVICE).eval()
print("Model loaded!")

# ── Step 4: Segmentation ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 4: Running segmentation")
print("="*50)

def preprocess(img):
    rgb = img[[2, 1, 0], :, :]
    p2  = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-9)
    rgb = rgb.clip(0, 1)
    rgb = (rgb - RGB_MEAN[:, None, None]) / RGB_STD[:, None, None]
    t   = torch.tensor(rgb).unsqueeze(0)
    t   = F.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)
    return t.to(DEVICE)

def segment(img):
    with torch.no_grad():
        logits = model(pixel_values=preprocess(img)).logits
        logits = F.interpolate(logits, size=(512,512),
                               mode='bilinear', align_corners=False)
        return logits.argmax(dim=1).squeeze().cpu().numpy()

seg_2015 = segment(img_2015)
seg_2024 = segment(img_2024_aligned)

print("2015 class distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:12s}: {(seg_2015==i).mean()*100:.1f}%")

print("\n2024 class distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:12s}: {(seg_2024==i).mean()*100:.1f}%")

# ── Step 5: Change detection ──────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 5: Computing change map")
print("="*50)

veg_to_bld = (seg_2015 == 3) & (seg_2024 == 1)
bg_to_bld  = (seg_2015 == 0) & (seg_2024 == 1)
bld_to_veg = (seg_2015 == 1) & (seg_2024 == 3)
bld_to_bg  = (seg_2015 == 1) & (seg_2024 == 0)
any_change = (seg_2015 != seg_2024)

print(f"Total changed       : {any_change.mean()*100:.1f}%")
print(f"Vegetation→Building : {veg_to_bld.sum():,} pixels")
print(f"Background→Building : {bg_to_bld.sum():,} pixels")
print(f"Building→Vegetation : {bld_to_veg.sum():,} pixels")
print(f"Building→Background : {bld_to_bg.sum():,} pixels")

# ── Step 6: Save results ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 6: Saving results to GCS")
print("="*50)

save_npy(any_change.astype(np.uint8), f"{GCS_OUT}/change_map.npy")
save_npy(seg_2015.astype(np.uint8),   f"{GCS_OUT}/seg_2015.npy")
save_npy(seg_2024.astype(np.uint8),   f"{GCS_OUT}/seg_2024.npy")

print("\nPhase 3 complete!")
print(f"Results → gs://{GCS_BUCKET}/{GCS_OUT}/")