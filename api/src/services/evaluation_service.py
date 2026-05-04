import re
from typing import Any


def _normalize_ref(row: dict[str, Any]) -> tuple[str, str, int | None]:
    doc_id = str(row.get("doc_id") or "").strip()
    section_id = str(row.get("section_id") or "").strip()
    chunk_index_raw = row.get("chunk_index")
    chunk_index: int | None
    if chunk_index_raw is None:
        chunk_index = None
    else:
        try:
            chunk_index = int(chunk_index_raw)
        except (TypeError, ValueError):
            chunk_index = None
    return (doc_id, section_id, chunk_index)


def retrieval_recall_at_k(
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    k: int,
) -> dict[str, Any] | None:
    """Reranked chunks as oracle relevance; hybrid top-k as retrieved set."""
    if k <= 0:
        return None
    gt = {_normalize_ref(x) for x in reranked_results if x}
    if not gt:
        return None
    cand = hybrid_results[:k]
    got = {_normalize_ref(x) for x in cand}
    hits = gt.intersection(got)
    return {
        "metric": "retrieval_recall@k",
        "k": int(k),
        "relevant_total": len(gt),
        "hits": len(hits),
        "score": len(hits) / len(gt),
        "matched_references": [
            {"doc_id": d, "section_id": s, "chunk_index": c} for d, s, c in sorted(hits)
        ],
    }


def citation_coverage(answer: str, context_chunk_count: int) -> dict[str, Any] | None:
    if context_chunk_count <= 0:
        return None
    cited = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)]
    valid = [i for i in cited if 1 <= i <= context_chunk_count]
    invalid_count = sum(1 for i in cited if i < 1 or i > context_chunk_count)
    unique_valid = len(set(valid))
    return {
        "context_chunk_count": context_chunk_count,
        "cited_positions": cited,
        "valid_count": len(valid),
        "invalid_count": invalid_count,
        "unique_valid_positions": unique_valid,
        "position_coverage": unique_valid / context_chunk_count,
    }


def build_query_metrics(
    *,
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    k: int,
    answer: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rr = retrieval_recall_at_k(hybrid_results, reranked_results, k)
    if rr is not None:
        out["retrieval_recall_at_k"] = rr
    cc = citation_coverage(answer, len(reranked_results))
    if cc is not None:
        out["citations"] = cc
    return out
