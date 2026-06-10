"""
main.py
-------
GeoVision QA — FastAPI application with Langfuse observability.

Every request creates a Langfuse trace so you can inspect the full pipeline:
  input image → RAG retrieval → SegFormer → LLaVA → answer

Endpoints:
  GET  /health          Health check + model status
  GET  /rag-context     Inspect the RAG knowledge base
  POST /segment         Semantic segmentation of an uploaded image
  POST /change-detect   Change detection between two uploaded images
  POST /vqa             Visual QA (LLaVA + RAG)
  POST /analyze         Full pipeline: segment + VQA in one call
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import answer_question, detect_changes, segment
from .model_loader import ModelRegistry, cleanup_models, load_all_models
from .observability import end_trace, get_langfuse, start_trace
from .rag_context import build_rag_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

GCS_BUCKET = os.getenv("GCS_BUCKET", "geovision-data")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "20")) * 1024 * 1024

_registry: ModelRegistry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry
    logger.info("=== GeoVision API starting up ===")
    _registry = load_all_models(GCS_BUCKET)
    # Pre-warm the RAG context cache
    try:
        build_rag_context(GCS_BUCKET)
        logger.info("RAG context cache warmed.")
    except Exception:
        logger.warning("RAG context pre-warm failed — will retry on first request.")
    logger.info("=== Models loaded — ready to serve ===")
    yield
    logger.info("=== GeoVision API shutting down ===")
    cleanup_models(_registry)
    get_langfuse().flush()


app = FastAPI(
    title="GeoVision QA API",
    description=(
        "Satellite image analysis: segmentation, change detection, "
        "and visual QA over Sentinel-2 imagery. "
        "All requests are traced in Langfuse."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

async def _read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ("image/png", "image/jpeg", "image/tiff", "image/geotiff"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {upload.content_type}",
        )
    data = await upload.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")


def _mask_to_list(mask: np.ndarray) -> list:
    return mask.tolist()


def _request_id() -> str:
    return str(uuid.uuid4())[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models": {
            "segformer": _registry.segformer is not None,
            "llava": _registry.llava is not None,
        },
        "device": _registry.device,
        "gcs_bucket": GCS_BUCKET,
    }


@app.get("/rag-context", tags=["System"])
async def rag_context() -> dict[str, str]:
    try:
        return {"context": build_rag_context(GCS_BUCKET)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/segment", tags=["Segmentation"])
async def segment_image(
    image: Annotated[UploadFile, File()],
    include_mask: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    """Semantic segmentation — returns per-class pixel % breakdown."""
    if _registry.segformer is None:
        raise HTTPException(status_code=503, detail="SegFormer not loaded.")

    img = await _read_image(image)
    req_id = _request_id()

    trace = start_trace(
        name="segment",
        session_id=req_id,
        tags=["geovision", "segmentation"],
        input={"filename": image.filename, "width": img.width, "height": img.height},
    )

    try:
        result = segment(img, _registry, trace=trace)
    except Exception as exc:
        end_trace(trace, error=str(exc))
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    response: dict[str, Any] = {
        "request_id": req_id,
        "image_size": {"width": img.width, "height": img.height},
        "distribution": result["distribution"],
    }
    if include_mask:
        response["mask"] = _mask_to_list(result["mask"])

    end_trace(trace, output={"distribution": result["distribution"]})
    return response


@app.post("/change-detect", tags=["Change Detection"])
async def change_detection(
    image_before: Annotated[UploadFile, File()],
    image_after: Annotated[UploadFile, File()],
    include_masks: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    """Change detection between two satellite images."""
    if _registry.segformer is None:
        raise HTTPException(status_code=503, detail="SegFormer not loaded.")

    img_before = await _read_image(image_before)
    img_after = await _read_image(image_after)
    req_id = _request_id()

    trace = start_trace(
        name="change-detect",
        session_id=req_id,
        tags=["geovision", "change-detection"],
        input={
            "before": image_before.filename,
            "after": image_after.filename,
        },
    )

    try:
        result = detect_changes(img_before, img_after, _registry, trace=trace)
    except Exception as exc:
        end_trace(trace, error=str(exc))
        logger.exception("Change detection failed")
        raise HTTPException(status_code=500, detail=str(exc))

    response: dict[str, Any] = {
        "request_id": req_id,
        "changed_pct": result["changed_pct"],
        "distribution_before": result["dist_before"],
        "distribution_after": result["dist_after"],
        "change_types": result["change_types"],
    }
    if include_masks:
        response["mask_before"] = _mask_to_list(result["mask_before"])
        response["mask_after"] = _mask_to_list(result["mask_after"])
        response["change_map"] = _mask_to_list(result["change_map"].astype(np.uint8))

    end_trace(trace, output={
        "changed_pct": result["changed_pct"],
        "change_types": result["change_types"],
    })
    return response


@app.post("/vqa", tags=["VQA"])
async def visual_qa(
    image: Annotated[UploadFile, File()],
    question: Annotated[str, Form()],
    use_rag: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    """
    Visual QA: answer a natural language question about a satellite image.
    Uses LLaVA-1.5-7B (QLoRA fine-tuned) + RAG context from Phase 3 stats.
    """
    if _registry.llava is None:
        raise HTTPException(status_code=503, detail="LLaVA not loaded.")

    img = await _read_image(image)
    req_id = _request_id()

    trace = start_trace(
        name="vqa",
        session_id=req_id,
        tags=["geovision", "vqa", "rag" if use_rag else "no-rag"],
        input={
            "question": question,
            "filename": image.filename,
            "use_rag": use_rag,
        },
    )

    try:
        result = answer_question(img, question, _registry, GCS_BUCKET, use_rag, trace=trace)
    except Exception as exc:
        end_trace(trace, error=str(exc))
        logger.exception("VQA inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    response = {
        "request_id": req_id,
        "question": result["question"],
        "answer": result["answer"],
        "rag_used": result["rag_used"],
    }
    end_trace(trace, output={"answer": result["answer"]})
    return response


@app.post("/analyze", tags=["Full Pipeline"])
async def analyze(
    image: Annotated[UploadFile, File()],
    question: Annotated[str, Form()] = (
        "What land cover types are present and what do they tell us about this area?"
    ),
    use_rag: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    """Full pipeline: segmentation + VQA in one request."""
    img = await _read_image(image)
    req_id = _request_id()

    trace = start_trace(
        name="analyze",
        session_id=req_id,
        tags=["geovision", "full-pipeline"],
        input={"question": question, "filename": image.filename},
    )

    response: dict[str, Any] = {"request_id": req_id}

    if _registry.segformer is not None:
        try:
            seg = segment(img, _registry, trace=trace)
            response["segmentation"] = {"distribution": seg["distribution"]}
        except Exception as exc:
            response["segmentation"] = {"error": str(exc)}
    else:
        response["segmentation"] = {"error": "SegFormer not loaded"}

    if _registry.llava is not None:
        try:
            vqa = answer_question(img, question, _registry, GCS_BUCKET, use_rag, trace=trace)
            response["vqa"] = {
                "question": vqa["question"],
                "answer": vqa["answer"],
                "rag_used": vqa["rag_used"],
            }
        except Exception as exc:
            response["vqa"] = {"error": str(exc)}
    else:
        response["vqa"] = {"error": "LLaVA not loaded"}

    end_trace(trace, output=response)
    return response
