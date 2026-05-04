from pydantic import BaseModel, Field


class QueryDto(BaseModel):
    query: str = Field(..., description="Question to answer with retrieval-augmented generation.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many reranked chunks to use as context.")
    doc_id: str | None = Field(default=None, description="Optional filter to one indexed doc_id.")
    hybrid: bool = Field(default=True, description="Enable BM25+RRF hybrid retrieval before generation.")
    rrf_k: int = Field(default=60, ge=1, le=500)
    rrf_k_bm25: int | None = Field(default=None, ge=4, le=200)
    cross_encoder: bool = Field(default=True, description="Rerank with cross-encoder before generation.")
    rerank_pool: int = Field(default=40, ge=1, le=200)
    candidate_multiplier: int = Field(default=8, ge=2, le=30)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=600, ge=64, le=4000)
    metrics: bool = Field(
        default=False,
        description="If true, response includes retrieval and citation diagnostics (no extra UI labels).",
    )
