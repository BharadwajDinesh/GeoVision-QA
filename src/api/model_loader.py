"""
model_loader.py
---------------
Loads and caches all models required by the GeoVision API:

  1. SegFormer-b2  — semantic segmentation (Phase 2 checkpoint)
  2. LLaVA-1.5-7B  — visual question answering (Phase 4 LoRA adapter)

Models are downloaded from GCS on first load, then kept in memory.
Loading is triggered once at FastAPI startup (lifespan event) so the
first request is never slow.

GCS layout expected:
  gs://<bucket>/geovision/phase2/checkpoints/best_segformer.pt
  gs://<bucket>/geovision/phase4/checkpoints/best_llava_rsvqa/
      adapter_config.json
      adapter_model.safetensors
      tokenizer.json  ...
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from google.cloud import storage
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("GCS_BUCKET", "geovision-data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── SegFormer config (must match Phase 2 training) ───────────────────────────
SEGFORMER_BASE = "nvidia/segformer-b2-finetuned-ade-512-512"
NUM_CLASSES = 5
ID2LABEL = {0: "background", 1: "building", 2: "road", 3: "vegetation", 4: "water"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── LLaVA config ─────────────────────────────────────────────────────────────
LLAVA_BASE = "llava-hf/llava-1.5-7b-hf"


@dataclass
class ModelRegistry:
    """Container for all loaded models — passed through FastAPI app state."""
    segformer: Optional[SegformerForSemanticSegmentation] = None
    seg_processor: Optional[SegformerImageProcessor] = None
    llava: Optional[LlavaForConditionalGeneration] = None
    llava_processor: Optional[AutoProcessor] = None
    device: str = DEVICE
    adapter_local_dir: Optional[Path] = field(default=None, repr=False)


# ── GCS helpers ──────────────────────────────────────────────────────────────

def _gcs_client() -> storage.Client:
    return storage.Client()


def _download_blob_to_bytes(bucket: str, blob_path: str) -> bytes:
    client = _gcs_client()
    return client.bucket(bucket).blob(blob_path).download_as_bytes()


def _download_directory_from_gcs(bucket_name: str, gcs_prefix: str, local_dir: Path) -> None:
    """
    Recursively download every blob under gcs_prefix into local_dir,
    preserving the directory structure relative to the prefix.
    """
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))

    if not blobs:
        raise FileNotFoundError(
            f"No blobs found under gs://{bucket_name}/{gcs_prefix}"
        )

    for blob in blobs:
        relative_path = blob.name[len(gcs_prefix):].lstrip("/")
        if not relative_path:          # skip the directory placeholder blob
            continue
        dest = local_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("  ↓ %s", blob.name)
        blob.download_to_filename(str(dest))


# ── SegFormer loader ─────────────────────────────────────────────────────────

def load_segformer(bucket_name: str = BUCKET_NAME) -> tuple[
    SegformerForSemanticSegmentation, SegformerImageProcessor
]:
    """
    1. Build SegFormer-b2 architecture with 5 output classes.
    2. Download fine-tuned checkpoint from GCS.
    3. Load weights, set eval mode, move to device.
    """
    logger.info("Loading SegFormer checkpoint from GCS …")

    ckpt_bytes = _download_blob_to_bytes(
        bucket_name,
        "geovision/phase2/checkpoints/best_segformer.pt",
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_BASE,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    state_dict = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    # Strip 'module.' prefix if the checkpoint was saved from DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(DEVICE)

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_BASE)

    logger.info("SegFormer ready on %s.", DEVICE)
    return model, processor


# ── LLaVA loader ─────────────────────────────────────────────────────────────

def load_llava(bucket_name: str = BUCKET_NAME) -> tuple[
    LlavaForConditionalGeneration, AutoProcessor, Path
]:
    """
    1. Download LoRA adapter directory from GCS to a temp directory.
    2. Load LLaVA-1.5-7B base in 8-bit quantization.
    3. Attach the LoRA adapter via PEFT.
    4. Return model, processor, and the temp dir path (for cleanup on shutdown).
    """
    logger.info("Downloading LLaVA LoRA adapter from GCS …")

    adapter_dir = Path(tempfile.mkdtemp(prefix="llava_adapter_"))
    _download_directory_from_gcs(
        bucket_name,
        "geovision/phase4/checkpoints/best_llava_rsvqa",
        adapter_dir,
    )
    logger.info("Adapter downloaded to %s", adapter_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    logger.info("Loading LLaVA-1.5-7B base model (8-bit) …")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info("Attaching LoRA adapter …")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    processor = AutoProcessor.from_pretrained(LLAVA_BASE)

    logger.info("LLaVA ready.")
    return model, processor, adapter_dir


# ── Top-level loader called at startup ───────────────────────────────────────

def load_all_models(bucket_name: str = BUCKET_NAME) -> ModelRegistry:
    """
    Load every model and return a populated ModelRegistry.
    Called once during FastAPI lifespan startup.
    """
    registry = ModelRegistry(device=DEVICE)

    try:
        registry.segformer, registry.seg_processor = load_segformer(bucket_name)
    except Exception:
        logger.exception("SegFormer failed to load — segmentation endpoints will be unavailable.")

    try:
        registry.llava, registry.llava_processor, registry.adapter_local_dir = (
            load_llava(bucket_name)
        )
    except Exception:
        logger.exception("LLaVA failed to load — VQA endpoints will be unavailable.")

    return registry


def cleanup_models(registry: ModelRegistry) -> None:
    """Free GPU memory and remove temp files on shutdown."""
    logger.info("Cleaning up models …")
    del registry.segformer
    del registry.llava
    torch.cuda.empty_cache()

    if registry.adapter_local_dir and registry.adapter_local_dir.exists():
        shutil.rmtree(registry.adapter_local_dir, ignore_errors=True)
        logger.info("Removed adapter temp dir %s", registry.adapter_local_dir)
