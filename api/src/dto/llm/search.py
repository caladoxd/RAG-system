from pydantic import BaseModel, Field


class SearchDto(BaseModel):
    query: str = Field(..., description="Natural language query.")
    top_k: int = Field(default=5, ge=1, le=100)
    doc_id: str | None = Field(default=None, description="Optional filter to one indexed doc_id.")
    hybrid: bool = Field(
        default=True,
        description="If true: Milvus candidates, asymmetric BM25+RRF (lexical leg uses a tighter k), optional cross-encoder rerank.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=500,
        description="RRF k for the vector-similarity rank leg (larger = flatter ranks).",
    )
    rrf_k_bm25: int | None = Field(
        default=None,
        ge=4,
        le=200,
        description="RRF k for the BM25 rank leg; omit to auto-scale from rrf_k (smaller than vector k so lexical order matters more).",
    )
    cross_encoder: bool = Field(
        default=True,
        description="If true (and hybrid): rerank top rerank_pool RRF results with a cross-encoder, return top_k.",
    )
    rerank_pool: int = Field(
        default=40,
        ge=1,
        le=200,
        description="How many best RRF hits to score with the cross-encoder (ignored when cross_encoder is false).",
    )
    candidate_multiplier: int = Field(
        default=8,
        ge=2,
        le=30,
        description="Milvus ANN retrieves min(300, top_k * candidate_multiplier) chunks before BM25+RRF.",
    )
