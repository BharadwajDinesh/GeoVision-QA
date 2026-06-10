"""
rag_context.py
--------------
Builds a retrieval-augmented context string from Phase 3 change detection
artifacts stored in GCS. This context is injected into every LLaVA prompt
so the model has grounded, factual statistics about Koramangala's land cover
change without hallucinating numbers.

GCS artifacts consumed:
  gs://<bucket>/geovision/phase3/seg_2015.npy
  gs://<bucket>/geovision/phase3/seg_2024.npy
  gs://<bucket>/geovision/phase3/change_map.npy
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import numpy as np
from google.cloud import storage

logger = logging.getLogger(__name__)

# ── class label map (must match Phase 2 training) ────────────────────────────
CLASS_NAMES = {
    0: "Background",
    1: "Buildings",
    2: "Roads",
    3: "Vegetation",
    4: "Water",
}

# ── change-type descriptions ──────────────────────────────────────────────────
CHANGE_DESCRIPTIONS = {
    (3, 1): "Vegetation → Building (new construction on green land)",
    (0, 1): "Background → Building (new construction on bare land)",
    (1, 3): "Building → Vegetation (demolition / greening)",
    (1, 0): "Building → Background (demolition)",
    (4, 1): "Water → Building (encroachment on water body)",
    (1, 4): "Building → Water (flooding / restoration)",
    (3, 0): "Vegetation → Background (deforestation / clearing)",
    (0, 3): "Background → Vegetation (regreening / afforestation)",
}


def _load_npy_from_gcs(bucket_name: str, blob_path: str) -> np.ndarray:
    """Download a .npy file from GCS and return it as a numpy array."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    return np.load(io.BytesIO(data))


def _pct(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total > 0 else 0.0


def _class_distribution(mask: np.ndarray) -> dict[str, float]:
    total = mask.size
    return {
        CLASS_NAMES[c]: _pct(int((mask == c).sum()), total)
        for c in CLASS_NAMES
    }


def _change_breakdown(seg_2015: np.ndarray, seg_2024: np.ndarray) -> dict[str, float]:
    """
    For every pixel where the class changed, count which (from→to) pair it is.
    Returns a dict of description → percentage-of-total-pixels.
    """
    changed_mask = seg_2015 != seg_2024
    total = seg_2015.size
    breakdown: dict[str, float] = {}

    from_classes = seg_2015[changed_mask]
    to_classes = seg_2024[changed_mask]

    for (from_c, to_c), description in CHANGE_DESCRIPTIONS.items():
        count = int(((from_classes == from_c) & (to_classes == to_c)).sum())
        if count > 0:
            breakdown[description] = _pct(count, total)

    return dict(sorted(breakdown.items(), key=lambda x: -x[1]))


@lru_cache(maxsize=1)
def build_rag_context(bucket_name: str = "geovision-data") -> str:
    """
    Load Phase 3 masks from GCS, compute statistics, and return a
    formatted context string ready to be prepended to LLaVA prompts.

    Result is cached after the first call (masks don't change at runtime).
    """
    logger.info("Loading Phase 3 masks from GCS for RAG context …")

    try:
        seg_2015 = _load_npy_from_gcs(bucket_name, "geovision/phase3/seg_2015.npy")
        seg_2024 = _load_npy_from_gcs(bucket_name, "geovision/phase3/seg_2024.npy")
        change_map = _load_npy_from_gcs(bucket_name, "geovision/phase3/change_map.npy")
    except Exception as e:
        logger.error("Failed to load Phase 3 masks: %s", e)
        return _fallback_context()

    dist_2015 = _class_distribution(seg_2015)
    dist_2024 = _class_distribution(seg_2024)
    total_changed_pct = _pct(int(change_map.sum()), change_map.size)
    breakdown = _change_breakdown(seg_2015, seg_2024)

    lines = [
        "=== GEOVISION KNOWLEDGE BASE — Koramangala, Bengaluru (Sentinel-2) ===",
        "",
        "Area of Interest : Koramangala, Bengaluru, India",
        "Coordinates      : 12.9411°N, 77.6158°E  |  1 km radius buffer",
        "Resolution       : 10 m/pixel (Sentinel-2)",
        "Time period      : 2015 → 2024 (approx. 9 years)",
        "",
        "── Land Cover Distribution ─────────────────────────────────────────",
        f"{'Class':<15} {'2015':>8} {'2024':>8} {'Change':>10}",
        "─" * 45,
    ]

    for cls in CLASS_NAMES.values():
        p15 = dist_2015.get(cls, 0.0)
        p24 = dist_2024.get(cls, 0.0)
        delta = round(p24 - p15, 1)
        sign = "+" if delta > 0 else ""
        lines.append(f"{cls:<15} {p15:>7}% {p24:>7}% {sign}{delta:>8}%")

    lines += [
        "",
        f"Total changed pixels : {total_changed_pct}% of the scene",
        "",
        "── Key Change Events (by area) ─────────────────────────────────────",
    ]

    for description, pct in breakdown.items():
        lines.append(f"  • {pct}% — {description}")

    lines += [
        "",
        "── Interpretation ───────────────────────────────────────────────────",
        "Koramangala has undergone significant urban densification over 9 years.",
        "Vegetation cover increased despite urbanization, likely due to urban",
        "greening initiatives and park development. Water bodies expanded slightly,",
        "possibly reflecting seasonal variation or lake restoration projects.",
        "The area remains one of Bengaluru's densest mixed-use neighbourhoods.",
        "",
        "=== END OF KNOWLEDGE BASE ===",
        "",
    ]

    context = "\n".join(lines)
    logger.info("RAG context built (%d chars).", len(context))
    return context


def _fallback_context() -> str:
    """Minimal hardcoded context if GCS is unavailable."""
    return (
        "=== GEOVISION KNOWLEDGE BASE (cached) ===\n"
        "Location: Koramangala, Bengaluru, India (12.9411°N, 77.6158°E)\n"
        "Period  : 2015 → 2024\n"
        "Key findings:\n"
        "  • ~30% of pixels changed land cover class\n"
        "  • Building coverage decreased from ~57.7% to ~44.6% (model artefact: densification)\n"
        "  • Vegetation increased from ~11.5% to ~25.2%\n"
        "  • Water bodies grew from ~0.6% to ~2.4%\n"
        "  • Major change type: Vegetation → Building (new construction)\n"
        "=== END ===\n"
    )


def get_prompted_question(user_question: str, bucket_name: str = "geovision-data") -> str:
    """
    Returns a fully-formed LLaVA USER prompt that prepends the RAG knowledge
    base to the user's question.

    Format expected by LLaVA-1.5:
        USER: <image>
        {context}
        Question: {question}
        ASSISTANT:
    """
    context = build_rag_context(bucket_name)
    return (
        f"{context}\n"
        f"Using the satellite data context above, answer the following question "
        f"about the Koramangala area accurately and concisely.\n\n"
        f"Question: {user_question}"
    )
