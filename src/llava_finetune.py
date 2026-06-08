"""
Phase 4 — LLaVA-1.5 QLoRA Fine-tuning on RSVQA-LR
-----------------------------------------------------
Fine-tunes LLaVA-1.5-7B with 4-bit QLoRA on the RSVQA-LR satellite
image Q&A dataset. Designed to run on Google Colab T4 (16GB VRAM).

Memory budget (approx):
    LLaVA-1.5-7B 4-bit : ~5 GB
    LoRA adapters       : ~0.3 GB
    Activations (bs=4)  : ~7 GB
    Overhead            : ~3 GB
    ─────────────────────────────
    Total               : ~15.3 GB  ← fits T4

Usage (Colab):
    !python src/llava_finetune.py \
        --bucket geovision-data \
        --output-dir gs://geovision-data/geovision/phase4/checkpoints \
        --epochs 3 \
        --batch-size 4
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import Dataset
from google.cloud import storage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)


# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL  = "llava-hf/llava-1.5-7b-hf"
GCS_PREFIX  = "geovision/phase4/rsvqa"

LORA_CONFIG = LoraConfig(
    r=16,                        # rank — higher = more capacity, more VRAM
    lora_alpha=32,
    target_modules=[             # apply LoRA to language model attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ── GCS helpers ───────────────────────────────────────────────────────────────

class GCSImageCache:
    """
    Lazily downloads satellite images from GCS to a local cache dir.
    Avoids re-downloading the same image on every epoch.
    """
    def __init__(self, bucket_name: str, gcs_prefix: str, cache_dir: Path):
        self.client     = storage.Client()
        self.bucket     = self.client.bucket(bucket_name)
        self.gcs_prefix = gcs_prefix
        self.cache_dir  = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, relative_path: str) -> Path:
        """Return local path to image, downloading from GCS if needed."""
        local_path = self.cache_dir / relative_path
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob_name = f"{self.gcs_prefix}/{relative_path}"
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
        return local_path


def download_jsonl(bucket_name: str, gcs_path: str, local_path: Path) -> None:
    """Download a JSONL file from GCS."""
    if local_path.exists():
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    print(f"  Downloaded {gcs_path} → {local_path}")


def upload_checkpoint(local_dir: Path, bucket_name: str, gcs_prefix: str) -> None:
    """Upload a checkpoint directory to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for f in local_dir.rglob("*"):
        if f.is_file():
            blob_name = f"{gcs_prefix}/{f.relative_to(local_dir)}"
            bucket.blob(blob_name).upload_from_filename(str(f))
    print(f"  Checkpoint uploaded → gs://{bucket_name}/{gcs_prefix}/")


# ── Dataset ───────────────────────────────────────────────────────────────────

class RSVQADataset(torch.utils.data.Dataset):
    """
    Streams RSVQA-LR samples in LLaVA conversation format.
    Loads images lazily from local GCS cache.
    """

    IMAGE_TOKEN_ID = 32000   # <image> token in LLaVA-1.5 vocabulary

    def __init__(
        self,
        jsonl_path: Path,
        processor: AutoProcessor,
        image_cache: GCSImageCache,
        max_length: int = 256,
    ):
        self.processor   = processor
        self.image_cache = image_cache
        self.max_length  = max_length
        self.samples     = self._load(jsonl_path)

    def _load(self, path: Path) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"  Loaded {len(samples):,} samples from {path.name}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        convs  = sample["conversations"]

        # Format prompt manually — apply_chat_template strips the <image> token
        # causing "image tokens: 0, features: N" mismatch at training time.
        # LLaVA-1.5 expects: "USER: <image>\n{question} ASSISTANT: {answer}"
        question = convs[0]["value"]  # already contains "<image>\n..."
        answer   = convs[1]["value"]
        prompt   = f"USER: {question} ASSISTANT: {answer}"

        # Load image
        img_path = self.image_cache.get(sample["image"])
        image    = Image.open(img_path).convert("RGB")

        # Tokenize
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)

        # Labels: mask prompt tokens, only supervise the answer
        # Find where the assistant turn starts by locating the last
        # human turn end token
        labels = input_ids.clone()
        answer_start = self._find_answer_start(input_ids)
        labels[:answer_start] = -100   # ignore loss on prompt

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
        }

    def _find_answer_start(self, input_ids: torch.Tensor) -> int:
        """
        Find the token index where the assistant answer begins.
        Looks for the ASSISTANT: marker token sequence.
        """
        # Heuristic: last occurrence of 319 (A) + 1 (the answer)
        # More robust: search for the separator token
        ids = input_ids.tolist()
        # LLaVA-1.5 uses "ASSISTANT:" as turn separator
        assistant_tokens = self.processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )
        n = len(assistant_tokens)
        for i in range(len(ids) - n, -1, -1):
            if ids[i:i + n] == assistant_tokens:
                return i + n
        return len(ids) // 2   # fallback: supervise second half


# ── Model setup ───────────────────────────────────────────────────────────────

def load_model_and_processor(base_model: str):
    """Load LLaVA-1.5 in 4-bit with QLoRA adapters."""
    print(f"\n  Loading processor from {base_model} …")
    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer.padding_side = "right"

    print(f"  Loading model in 4-bit NF4 …")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=BNBCONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training (freezes non-LoRA weights, casts norms)
    model = prepare_model_for_kbit_training(model)

    print("  Attaching LoRA adapters …")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    return model, processor


# ── Data collator ─────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors from a list of dataset items."""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "pixel_values":   torch.stack([b["pixel_values"]    for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Exact-match accuracy over decoded predictions vs labels."""
    predictions, labels = eval_pred
    # Predictions are logits — take argmax
    pred_ids  = predictions.argmax(-1)
    # Replace -100 in labels with pad token
    labels[labels == -100] = 0
    return {"exact_match": float((pred_ids == labels).all(-1).mean())}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    local_data = Path("data/llava_format")
    cache_dir  = Path("data/image_cache")
    ckpt_dir   = Path("checkpoints/llava_rsvqa")

    # ── Download JSONL files from GCS ────────────────────────────────────────
    print("\n[1/4] Fetching training data from GCS …")
    download_jsonl(args.bucket, f"{GCS_PREFIX}/train.jsonl", local_data / "train.jsonl")
    download_jsonl(args.bucket, f"{GCS_PREFIX}/val.jsonl",   local_data / "val.jsonl")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[2/4] Loading model …")
    model, processor = load_model_and_processor(BASE_MODEL)

    # ── Build datasets ───────────────────────────────────────────────────────
    print("\n[3/4] Building datasets …")
    image_cache = GCSImageCache(args.bucket, GCS_PREFIX, cache_dir)

    train_dataset = RSVQADataset(
        local_data / "train.jsonl", processor, image_cache,
        max_length=args.max_length,
    )
    val_dataset = RSVQADataset(
        local_data / "val.jsonl", processor, image_cache,
        max_length=args.max_length,
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,       # effective batch = 16
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=True,                           # needed with 4-bit quant
        optim="paged_adamw_8bit",            # memory-efficient optimizer
        logging_dir=str(ckpt_dir / "logs"),
        logging_steps=10,
        report_to="none",                    # disable wandb
        dataloader_num_workers=2,
        remove_unused_columns=False,         # keep pixel_values
        label_names=["labels"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("\n[4/4] Starting training …")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    # ── Save & upload ─────────────────────────────────────────────────────────
    best_ckpt = ckpt_dir / "best_llava_rsvqa"
    model.save_pretrained(str(best_ckpt))
    processor.save_pretrained(str(best_ckpt))
    print(f"\n  Saved best checkpoint → {best_ckpt}")

    if not args.skip_upload:
        upload_checkpoint(
            best_ckpt, args.bucket,
            "geovision/phase4/checkpoints/best_llava_rsvqa",
        )

    print("\n✅ Fine-tuning complete.")
    print(f"   Checkpoint : gs://{args.bucket}/geovision/phase4/checkpoints/best_llava_rsvqa/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA-1.5 on RSVQA-LR")
    parser.add_argument("--bucket",      default="geovision-data")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--max-length",  type=int,   default=256)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    train(args)
