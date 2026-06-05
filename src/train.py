"""
train.py
--------
Fine-tunes SegFormer-b2 on satellite segmentation tiles stored in GCS.
Saves best checkpoint back to GCS based on validation mIoU.

Usage:
    python src/train.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation
import numpy as np
from google.cloud import storage
import io
import sys
sys.path.append('/home/bharathd7900/geovision/src')

from seg_dataset import get_dataloaders, CLASS_WEIGHTS, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET       = "geovision-data"
GCS_CHECKPOINT   = "geovision/phase2/checkpoints/best_segformer.pt"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS       = 20
LR               = 6e-5
BATCH_SIZE       = 4
LOCAL_CKPT_PATH  = "/tmp/best_segformer.pt"

print(f"Device     : {DEVICE}")
print(f"Epochs     : {NUM_EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader, val_loader = get_dataloaders(
    batch_size=BATCH_SIZE,
    use_rgb=True,
    val_split=0.2,
)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\nLoading SegFormer-b2...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model = model.to(DEVICE)
print("Model loaded!")

# ── Loss and Optimizer ────────────────────────────────────────────────────────
weights   = CLASS_WEIGHTS.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS
)

# ── Train and Val functions ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits  = nn.functional.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=images)
            logits  = nn.functional.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate([p.flatten() for p in all_preds])
    all_labels = np.concatenate([l.flatten() for l in all_labels])

    ious = []
    for c in range(NUM_CLASSES):
        intersection = ((all_preds == c) & (all_labels == c)).sum()
        union        = ((all_preds == c) | (all_labels == c)).sum()
        if union > 0:
            ious.append(intersection / union)
    miou = np.mean(ious) if ious else 0.0
    return total_loss / len(loader), miou


def save_checkpoint_to_gcs(local_path: str) -> None:
    """Upload checkpoint to GCS."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(GCS_CHECKPOINT)
    blob.upload_from_filename(local_path)
    print(f"Checkpoint saved → gs://{GCS_BUCKET}/{GCS_CHECKPOINT}")


# ── Run Training ──────────────────────────────────────────────────────────────
print("\nStarting training...\n")
best_miou  = 0.0
best_epoch = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss         = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_miou = val_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"mIoU: {val_miou:.4f}")

    if val_miou > best_miou:
        best_miou  = val_miou
        best_epoch = epoch
        torch.save(model.state_dict(), LOCAL_CKPT_PATH)
        save_checkpoint_to_gcs(LOCAL_CKPT_PATH)
        print(f"  ✅ Best model saved (mIoU={best_miou:.4f})")

print(f"\nTraining complete!")
print(f"Best mIoU : {best_miou:.4f} at epoch {best_epoch}")
print(f"Checkpoint: gs://{GCS_BUCKET}/{GCS_CHECKPOINT}")