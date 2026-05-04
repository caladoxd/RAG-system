from typing import Any

from pydantic import BaseModel, Field


class SearchResponse(BaseModel):
    hybrid_results: list[dict[str, Any]] = Field(
        ...,
        description="After vector retrieval: BM25+RRF-ranked full candidate list when hybrid; else all vector hits ordered.",
    )
    reranked_results: list[dict[str, Any]] = Field(
        ...,
        description="Final top_k: cross-encoder order when enabled; else top RRF (or vector) hits.",
    )
