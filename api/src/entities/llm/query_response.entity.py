from typing import Any

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer grounded on retrieved chunks.")
    context_results: list[dict[str, Any]] = Field(
        ...,
        description="Top reranked chunks used as generation context.",
    )
