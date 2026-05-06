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


class FaithfulnessMetric(BaseModel):
    metric: str = Field(default="faithfulness", description="RAGAS faithfulness (answer vs contexts).")
    score: float | None = Field(
        default=None,
        description="0–1 when scoring succeeded.",
    )
    error: str | None = Field(
        default=None,
        description="Set when the judge LLM call failed or dependencies are missing.",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Wall time for the faithfulness judge run (floored ms).",
    )


class LLMJudgeMetric(BaseModel):
    metric: str
    score: float | None = Field(default=None, description="0-1 when scoring succeeded.")
    error: str | None = Field(
        default=None,
        description="Set when the judge LLM call failed or dependencies are missing.",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Wall time for the judge run (floored ms).",
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
    context_relevance: LLMJudgeMetric | None = Field(
        default=None,
        description="RAGAS context relevance when metrics=true.",
    )
    faithfulness: FaithfulnessMetric | None = Field(
        default=None,
        description="RAGAS faithfulness when metrics=true (uses OPENAI_BASE_URL / RAGAS_EVAL_MODEL).",
    )
    answer_relevancy: LLMJudgeMetric | None = Field(
        default=None,
        description="RAGAS answer relevancy when metrics=true.",
    )
    latencies_ms: dict[str, int] | None = Field(
        default=None,
        description="Per-step wall times in floored whole milliseconds (retrieval, generation, metrics, handler).",
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
