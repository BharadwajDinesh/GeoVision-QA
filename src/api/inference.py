"""
inference.py
------------
All model inference logic with Langfuse observability instrumentation.

Three pipelines:
  1. segment(image, registry, trace)               → class mask + distribution
  2. detect_changes(img_a, img_b, registry, trace) → change map + stats
  3. answer_question(image, q, registry, trace)    → LLaVA answer + RAG context
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model_loader import ModelRegistry
from .observability import llava_generation, rag_span, segformer_span
from .rag_context import CHANGE_DESCRIPTIONS, CLASS_NAMES, get_prompted_question

logger = logging.getLogger(__name__)

TILE_SIZE = 512
MAX_NEW_TOKENS = 256
GCS_BUCKET = "geovision-data"


# ── No-op trace (used when no trace is passed in) ────────────────────────────
class _NoOpTrace:
    def span(self, **kwargs): return self
    def generation(self, **kwargs): return self
    def update(self, **kwargs): pass
    def end(self, **kwargs): pass

def _noop_trace(): return _NoOpTrace()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Semantic Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _run_segformer(image: Image.Image, registry: ModelRegistry) -> np.ndarray:
    w, h = image.size
    if w <= TILE_SIZE and h <= TILE_SIZE:
        return _infer_single_tile(image, registry)
    return _infer_tiled(image, registry)


def _infer_single_tile(image: Image.Image, registry: ModelRegistry) -> np.ndarray:
    inputs = registry.seg_processor(images=image, return_tensors="pt").to(registry.device)
    with torch.no_grad():
        logits = registry.segformer(**inputs).logits
    upsampled = F.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def _infer_tiled(image: Image.Image, registry: ModelRegistry, overlap: int = 64) -> np.ndarray:
    w, h = image.size
    img_np = np.array(image)
    logit_accum = np.zeros((5, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    stride = TILE_SIZE - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1, y1 = x, y
            x2, y2 = min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)
            tile = Image.fromarray(img_np[y1:y2, x1:x2])
            inputs = registry.seg_processor(images=tile, return_tensors="pt").to(registry.device)
            with torch.no_grad():
                logits = registry.segformer(**inputs).logits
            up = F.interpolate(logits, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
            logit_accum[:, y1:y2, x1:x2] += up.squeeze(0).cpu().numpy()
            count_map[y1:y2, x1:x2] += 1.0

    avg_logits = logit_accum / np.maximum(count_map[np.newaxis], 1.0)
    return avg_logits.argmax(axis=0).astype(np.uint8)


def _mask_to_stats(mask: np.ndarray) -> dict[str, float]:
    total = mask.size
    return {
        CLASS_NAMES[c]: round(100 * float((mask == c).sum()) / total, 2)
        for c in CLASS_NAMES
    }


def segment(
    image: Image.Image,
    registry: ModelRegistry,
    trace=None,
) -> dict[str, Any]:
    if registry.segformer is None:
        raise RuntimeError("SegFormer model is not loaded.")

    with segformer_span(trace or _noop_trace(), image.size) as span:
        mask = _run_segformer(image, registry)
        distribution = _mask_to_stats(mask)
        span.update(output={"distribution": distribution})

    return {"mask": mask, "distribution": distribution}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Change Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_changes(
    image_before: Image.Image,
    image_after: Image.Image,
    registry: ModelRegistry,
    trace=None,
) -> dict[str, Any]:
    if registry.segformer is None:
        raise RuntimeError("SegFormer model is not loaded.")

    if image_before.size != image_after.size:
        image_after = image_after.resize(image_before.size, Image.BILINEAR)

    with segformer_span(trace or _noop_trace(), image_before.size) as span:
        span.update(metadata={"pass": "before"})
        mask_before = _run_segformer(image_before, registry)

    with segformer_span(trace or _noop_trace(), image_after.size) as span:
        span.update(metadata={"pass": "after"})
        mask_after = _run_segformer(image_after, registry)

    change_map = mask_before != mask_after
    changed_pct = round(100 * float(change_map.sum()) / change_map.size, 2)

    total = mask_before.size
    from_c_arr = mask_before[change_map]
    to_c_arr = mask_after[change_map]
    change_types: dict[str, float] = {}
    for (fc, tc), desc in CHANGE_DESCRIPTIONS.items():
        count = int(((from_c_arr == fc) & (to_c_arr == tc)).sum())
        if count > 0:
            change_types[desc] = round(100 * count / total, 2)
    change_types = dict(sorted(change_types.items(), key=lambda x: -x[1]))

    if trace:
        try:
            trace.span(name="change-diff").end(output={
                "changed_pct": changed_pct,
                "change_types": change_types,
            })
        except Exception:
            pass

    return {
        "mask_before": mask_before,
        "mask_after": mask_after,
        "change_map": change_map,
        "changed_pct": changed_pct,
        "dist_before": _mask_to_stats(mask_before),
        "dist_after": _mask_to_stats(mask_after),
        "change_types": change_types,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Visual Question Answering (LLaVA + RAG)
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    image: Image.Image,
    question: str,
    registry: ModelRegistry,
    bucket_name: str = GCS_BUCKET,
    use_rag: bool = True,
    trace=None,
) -> dict[str, str]:
    if registry.llava is None or registry.llava_processor is None:
        raise RuntimeError("LLaVA model is not loaded.")

    # Step 1 — RAG context retrieval
    with rag_span(trace or _noop_trace(), bucket_name) as span:
        if use_rag:
            user_text = get_prompted_question(question, bucket_name)
        else:
            user_text = question
        span.update(
            input={"question": question, "use_rag": use_rag},
            output={"context_chars": len(user_text)},
        )

    # Step 2 — LLaVA generation
    prompt = f"USER: <image>\n{user_text}\nASSISTANT:"

    with llava_generation(trace or _noop_trace(), prompt=prompt) as gen:
        inputs = registry.llava_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(registry.device)

        with torch.no_grad():
            output_ids = registry.llava.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        answer = registry.llava_processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

        gen.end(output=answer)

    return {
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "rag_used": use_rag,
    }
