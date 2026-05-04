"""Cross-encoder (query, passage) scoring — used after BM25+RRF fusion."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None
_model_id: str | None = None


def model_name() -> str:
    return os.getenv(
        "CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


def _load_sync() -> Any:
    global _model, _model_id

    name = model_name()
    with _lock:
        if _model is not None and _model_id == name:
            return _model
        logger.info("Loading cross-encoder %r (first use may download weights)", name)
        max_len = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
        _model = CrossEncoder(name, max_length=max_len)
        _model_id = name
        return _model


def warm_up_cross_encoder() -> None:
    """Load cross-encoder weights during app startup so the first query is not slow."""
    _load_sync()


def score_query_passages_sync(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    model = _load_sync()
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "12000"))
    q = query.strip()
    pairs = [[q, (p or "")[:max_chars]] for p in passages]
    batch = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))
    raw = model.predict(pairs, batch_size=batch, show_progress_bar=False)
    return [float(x) for x in raw]
