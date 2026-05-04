from typing import Any

from pydantic import BaseModel, Field


class RetrievalRecallAtKMetric(BaseModel):
    metric: str = Field(default="retrieval_recall@k")
    k: int
    relevant_total: int
    hits: int
    score: float
    matched_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Reranked chunks (treated as oracle) that also appear in hybrid top-k.",
    )


class CitationCoverageMetric(BaseModel):
    context_chunk_count: int
    cited_positions: list[int] = Field(
        default_factory=list,
        description="All [n] indices parsed from the answer, in order of appearance.",
    )
    valid_count: int
    invalid_count: int
    unique_valid_positions: int
    position_coverage: float = Field(
        ...,
        description="Unique valid citation indices divided by context_chunk_count.",
    )


class QueryMetrics(BaseModel):
    retrieval_recall_at_k: RetrievalRecallAtKMetric | None = Field(
        default=None,
        description="Share of reranked context chunks found in the hybrid top-k list (pipeline diagnostic).",
    )
    citations: CitationCoverageMetric | None = Field(
        default=None,
        description="How many [n] citations in the answer map to context snippet positions.",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer grounded on retrieved chunks.")
    context_results: list[dict[str, Any]] = Field(
        ...,
        description="Top reranked chunks used as generation context.",
    )
    metrics: QueryMetrics | None = Field(
        default=None,
        description="Populated when the request sets metrics=true.",
    )
