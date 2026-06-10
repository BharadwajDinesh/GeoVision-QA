"""
observability.py
----------------
Langfuse integration for GeoVision QA.

Every API request gets its own Langfuse trace so you can inspect:
  • The input image metadata + user question
  • RAG context retrieval (what was injected into the prompt, how long it took)
  • SegFormer segmentation results + latency
  • LLaVA prompt (full text sent to model) + raw answer + latency
  • Any errors at any step

Traces are visible at https://cloud.langfuse.com (or your self-hosted instance).

Required environment variables:
  LANGFUSE_PUBLIC_KEY   — from your Langfuse project settings
  LANGFUSE_SECRET_KEY   — from your Langfuse project settings
  LANGFUSE_HOST         — default: https://cloud.langfuse.com
                          set to http://localhost:3000 for self-hosted

If keys are missing, a no-op client is used so the API still works.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# ── Langfuse client (lazy-init, singleton) ────────────────────────────────────
_langfuse_client = None


def get_langfuse():
    """
    Return the shared Langfuse client, initialising it on first call.
    Returns None (silently) if credentials are not configured.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Observability is disabled — API will still work normally."
        )
        _langfuse_client = _NoOpLangfuse()
        return _langfuse_client

    try:
        from langfuse import Langfuse  # type: ignore
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse observability enabled at %s", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    except ImportError:
        logger.warning("langfuse package not installed. pip install langfuse")
        _langfuse_client = _NoOpLangfuse()

    return _langfuse_client


# ── No-op client: same interface, does nothing ───────────────────────────────

class _NoOpSpan:
    def update(self, **kwargs): pass
    def end(self, **kwargs): pass
    def generation(self, **kwargs): return self
    def span(self, **kwargs): return self

class _NoOpTrace(_NoOpSpan):
    def span(self, **kwargs): return _NoOpSpan()
    def generation(self, **kwargs): return _NoOpSpan()

class _NoOpLangfuse:
    def trace(self, **kwargs): return _NoOpTrace()
    def flush(self): pass


# ── Trace factory ─────────────────────────────────────────────────────────────

def start_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    input: Any = None,
):
    """
    Create a new Langfuse trace for one API request.
    Returns a trace object — pass it down to span helpers below.
    """
    lf = get_langfuse()
    return lf.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
        tags=tags or ["geovision"],
        input=input,
    )


# ── Span helpers (one per pipeline step) ─────────────────────────────────────

@contextmanager
def rag_span(trace, bucket: str) -> Generator[Any, None, None]:
    """
    Context manager for the RAG retrieval step.
    Records: what context was retrieved and how long it took.
    """
    span = trace.span(name="rag-context-retrieval", metadata={"gcs_bucket": bucket})
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        span.end(metadata={"latency_ms": elapsed_ms})


@contextmanager
def segformer_span(trace, image_size: tuple[int, int]) -> Generator[Any, None, None]:
    """
    Context manager for SegFormer inference.
    Records: image dimensions, class distribution output, latency.
    """
    span = trace.span(
        name="segformer-inference",
        input={"image_width": image_size[0], "image_height": image_size[1]},
    )
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        span.end(metadata={"latency_ms": elapsed_ms})


@contextmanager
def llava_generation(trace, prompt: str, model_name: str = "llava-1.5-7b-lora") -> Generator[Any, None, None]:
    """
    Context manager for LLaVA inference — logged as a Langfuse Generation
    (special type for LLM calls, shows prompt / completion diff in the UI).
    Records: full prompt sent, raw answer, token counts if available, latency.
    """
    gen = trace.generation(
        name="llava-vqa",
        model=model_name,
        input=prompt,
    )
    t0 = time.perf_counter()
    try:
        yield gen
    finally:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        gen.end(metadata={"latency_ms": elapsed_ms})


def end_trace(trace, output: Any = None, error: Optional[str] = None) -> None:
    """Finalise the trace with the overall output or error."""
    try:
        trace.update(output=output)
        if error:
            trace.update(metadata={"error": error})
        get_langfuse().flush()
    except Exception:
        pass   # observability must never break the API
