# The functions were built using AI, checked for coherence and functionality but code clean up is needed.
import asyncio
import logging
import os
import re
import time
from typing import Any

from . import llm_service
from .embedding_service import EMBEDDING_DIM, embed_texts

logger = logging.getLogger(__name__)


class MilvusStoreError(Exception):
    """Raised when Milvus connection, schema, or insert/search fails."""


_collection_name = os.getenv("MILVUS_COLLECTION", "document_chunks")
_text_max_len = int(os.getenv("MILVUS_TEXT_MAX_LENGTH", "16384"))
_section_id_max_len = 256
_lock = asyncio.Lock()


def _milvus_config() -> tuple[str, str]:
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    return host, port


def _connect_sync() -> None:
    from pymilvus import connections

    host, port = _milvus_config()
    connections.connect(alias="default", host=host, port=port, timeout=10.0)


def connections_has_default() -> bool:
    from pymilvus import connections

    return connections.has_connection("default")


def _ensure_collection_sync() -> Any:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    )

    if not connections_has_default():
        _connect_sync()

    if utility.has_collection(_collection_name):
        col = Collection(_collection_name)
        col.load()
        return col

    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=512,
        ),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section_id", dtype=DataType.VARCHAR, max_length=_section_id_max_len),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_text_max_len),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="Chunked documents + vectors")
    col = Collection(name=_collection_name, schema=schema)
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    logger.info("Created Milvus collection %s (dim=%s)", _collection_name, EMBEDDING_DIM)
    return col


def _sanitize_chunk_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}:{index}"
    safe = re.sub(r"[^a-zA-Z0-9:_\-]", "_", raw)
    return safe[:512]


def _truncate_text(text: str) -> str:
    if len(text) <= _text_max_len:
        return text
    return text[: _text_max_len - 30] + "\n...[truncated]"


def _expr_doc_id(doc_id: str) -> str:
    safe = doc_id.replace("\\", "\\\\").replace('"', '\\"')
    return f'doc_id == "{safe}"'


def _resolve_section_id_for_ingest(doc_id: str, section_id: str | None) -> str:
    """Use explicit section when provided; otherwise default to doc_id for stable chunk grouping."""
    s = (section_id or "").strip()
    if s:
        return s[:_section_id_max_len]
    return doc_id.strip()[:_section_id_max_len]


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _milvus_scalar(hit: Any, field: str) -> Any:
    """Read a scalar from a pymilvus Hit (values live under entity, not top-level)."""
    try:
        ent = hit["entity"]
    except (KeyError, TypeError):
        ent = None
    if isinstance(ent, dict) and field in ent:
        return ent[field]
    try:
        return hit[field]
    except (KeyError, TypeError):
        return None


def _milvus_hit_to_candidate(hit: Any, distance: float) -> dict[str, Any]:
    doc_id = _milvus_scalar(hit, "doc_id")
    section_raw = _milvus_scalar(hit, "section_id")
    chunk_idx = _milvus_scalar(hit, "chunk_index")
    text_val = _milvus_scalar(hit, "text")
    sid = "" if section_raw is None else str(section_raw)
    return {
        "vector_distance": float(distance or 0.0),
        "doc_id": doc_id,
        "section_id": sid,
        "chunk_index": chunk_idx,
        "text": text_val,
    }


def _finalize_hits_vector_only(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Milvus order: lower vector_distance is better (COSINE distance)."""
    sorted_hits = sorted(hits, key=lambda h: float(h.get("vector_distance", 0.0)))
    out: list[dict[str, Any]] = []
    for h in sorted_hits[:top_k]:
        vd = float(h.get("vector_distance", 0.0))
        out.append(
            {
                "doc_id": h.get("doc_id"),
                "section_id": h.get("section_id"),
                "chunk_index": h.get("chunk_index"),
                "text": h.get("text"),
                "vector_distance": vd,
            }
        )
    return out


def _default_rrf_k_bm25(rrf_k_vector: int) -> int:
    """Smaller k for the BM25 RRF leg → top lexical ranks contribute more than vector alone."""
    k = int(rrf_k_vector)
    return max(10, min(48, (k * 9 + 19) // 20))


def _bm25_scores_for_hits(query: str, hits: list[dict[str, Any]]) -> list[float]:
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(h.get("text")) for h in hits]
    corpus = [c if c else ["_"] for c in corpus]
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [0.0] * len(hits)
    bm25 = BM25Okapi(corpus)
    return [float(x) for x in bm25.get_scores(q_tokens)]


def _bm25_rrf_rerank(
    query: str,
    hits: list[dict[str, Any]],
    rrf_k: int,
    rrf_k_bm25: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse vector rank and BM25 rank with RRF (hybrid_score), best first.

    Uses a smaller k for the BM25 leg than for the vector leg so lexical rank
    differences matter more than raw ANN order. When any hit has BM25 > 0,
    hits with BM25 == 0 get no BM25 RRF term (vector-only half).
    """
    n = len(hits)
    if n == 0:
        return []
    if n == 1:
        h = hits[0]
        bm = _bm25_scores_for_hits(query, hits)[0]
        return [
            {
                "doc_id": h.get("doc_id"),
                "section_id": h.get("section_id"),
                "chunk_index": h.get("chunk_index"),
                "text": h.get("text"),
                "vector_distance": float(h.get("vector_distance", 0.0)),
                "bm25_score": float(bm),
                "hybrid_score": 1.0,
            }
        ]

    bm25_raw = _bm25_scores_for_hits(query, hits)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return _finalize_hits_vector_only(hits, len(hits))

    order_vec = sorted(range(n), key=lambda i: float(hits[i].get("vector_distance", 0.0)))
    vec_rank = [0] * n
    for r, idx in enumerate(order_vec):
        vec_rank[idx] = r + 1

    order_bm = sorted(range(n), key=lambda i: bm25_raw[i], reverse=True)
    bm_rank = [0] * n
    for r, idx in enumerate(order_bm):
        bm_rank[idx] = r + 1

    rrf_k_vec = int(rrf_k)
    rrf_k_bm = int(rrf_k_bm25) if rrf_k_bm25 is not None else _default_rrf_k_bm25(rrf_k_vec)
    bm25_eps = 1e-12
    lexical_signal = max(bm25_raw, default=0.0) > bm25_eps

    rrf: list[float] = []
    for i in range(n):
        v_term = 1.0 / (rrf_k_vec + vec_rank[i])
        if bm25_raw[i] > bm25_eps:
            b_term = 1.0 / (rrf_k_bm + bm_rank[i])
        elif lexical_signal:
            b_term = 0.0
        else:
            b_term = 1.0 / (rrf_k_bm + bm_rank[i])
        rrf.append(v_term + b_term)
    order = sorted(range(n), key=lambda i: rrf[i], reverse=True)
    out: list[dict[str, Any]] = []
    for i in order:
        h = hits[i]
        out.append(
            {
                "doc_id": h.get("doc_id"),
                "section_id": h.get("section_id"),
                "chunk_index": h.get("chunk_index"),
                "text": h.get("text"),
                "vector_distance": float(h.get("vector_distance", 0.0)),
                "bm25_score": float(bm25_raw[i]),
                "hybrid_score": float(rrf[i]),
            }
        )
    return out


def _cross_encoder_rerank_rows_sync(query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    from . import reranker_service

    if not rows:
        return []
    texts = [str(r.get("text") or "") for r in rows]
    try:
        ce_scores = reranker_service.score_query_passages_sync(query, texts)
    except Exception as e:
        logger.warning("Cross-encoder rerank failed, using BM25+RRF order: %s", e)
        return rows[:top_k]

    order = sorted(range(len(rows)), key=lambda i: ce_scores[i], reverse=True)
    out: list[dict[str, Any]] = []
    for i in order[:top_k]:
        r = dict(rows[i])
        r["rerank_score"] = float(ce_scores[i])
        out.append(r)
    return out


def _delete_by_doc_sync(doc_id: str) -> None:
    from pymilvus import Collection, utility

    if not connections_has_default():
        _connect_sync()
    if not utility.has_collection(_collection_name):
        return
    col = Collection(_collection_name)
    col.load()
    col.delete(expr=_expr_doc_id(doc_id))
    col.flush()


def _insert_sync(doc_id: str, chunks: list[str], embeddings: list[list[float]], section_id: str) -> int:
    from pymilvus import Collection

    _delete_by_doc_sync(doc_id)
    col = _ensure_collection_sync()
    texts = [_truncate_text(c) for c in chunks]
    sid = (section_id or "")[:_section_id_max_len]
    rows = [
        {
            "chunk_id": _sanitize_chunk_id(doc_id, i),
            "doc_id": doc_id,
            "section_id": sid,
            "chunk_index": i,
            "text": texts[i],
            "embedding": embeddings[i],
        }
        for i in range(len(chunks))
    ]
    col.insert(rows)
    col.flush()
    return len(chunks)


def _empty_timings_ms() -> dict[str, float]:
    return {
        "milvus_ann_ms": 0.0,
        "milvus_hit_materialize_ms": 0.0,
        "bm25_rrf_ms": 0.0,
        "cross_encoder_rerank_ms": 0.0,
        "vector_finalize_ms": 0.0,
    }


def _search_sync(
    query_vector: list[float],
    query_text: str,
    top_k: int,
    doc_id: str | None,
    hybrid: bool,
    rrf_k: int,
    rrf_k_bm25: int | None,
    cross_encoder: bool,
    rerank_pool: int,
    candidate_multiplier: int,
    *,
    collect_timings: bool = False,
) -> dict[str, Any]:
    from pymilvus import Collection, utility

    timings_ms: dict[str, float] = _empty_timings_ms() if collect_timings else {}

    def _pack(out: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        if collect_timings:
            out = dict(out)
            out["timings_ms"] = timings_ms
        return out

    if not connections_has_default():
        _connect_sync()
    if not utility.has_collection(_collection_name):
        return _pack({"hybrid_results": [], "reranked_results": []})
    col = Collection(_collection_name)
    col.load()
    expr = _expr_doc_id(doc_id) if doc_id else None
    milvus_limit = top_k
    if hybrid:
        milvus_limit = min(300, max(top_k * candidate_multiplier, top_k + 5))
    try:
        t0 = time.perf_counter()
        res: Any = col.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=milvus_limit,
            expr=expr,
            output_fields=["doc_id", "section_id", "chunk_index", "text"],
        )
        if collect_timings:
            timings_ms["milvus_ann_ms"] = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        logger.warning("Milvus search failed: %s", e)
        return _pack({"hybrid_results": [], "reranked_results": []})
    t_hits = time.perf_counter()
    hits: list[dict[str, Any]] = []
    for hit_group in res:
        for hit in hit_group:
            dist = float(hit.get("distance") or hit.get("score") or 0.0)
            hits.append(_milvus_hit_to_candidate(hit, dist))
    if collect_timings:
        timings_ms["milvus_hit_materialize_ms"] = (time.perf_counter() - t_hits) * 1000.0
    if hybrid:
        t_bm = time.perf_counter()
        fused = _bm25_rrf_rerank(query_text, hits, rrf_k, rrf_k_bm25)
        if collect_timings:
            timings_ms["bm25_rrf_ms"] = (time.perf_counter() - t_bm) * 1000.0
        pool = fused[: max(rerank_pool, top_k)]
        if cross_encoder and pool:
            t_ce = time.perf_counter()
            reranked = _cross_encoder_rerank_rows_sync(query_text, pool, top_k)
            if collect_timings:
                timings_ms["cross_encoder_rerank_ms"] = (time.perf_counter() - t_ce) * 1000.0
            return _pack({"hybrid_results": fused, "reranked_results": reranked})
        if collect_timings:
            timings_ms["cross_encoder_rerank_ms"] = 0.0
        return _pack({"hybrid_results": fused, "reranked_results": fused[:top_k]})
    t_fin = time.perf_counter()
    vec_all = _finalize_hits_vector_only(hits, len(hits))
    vec_top = _finalize_hits_vector_only(hits, top_k)
    if collect_timings:
        timings_ms["vector_finalize_ms"] = (time.perf_counter() - t_fin) * 1000.0
        timings_ms["bm25_rrf_ms"] = 0.0
        timings_ms["cross_encoder_rerank_ms"] = 0.0
    return _pack({"hybrid_results": vec_all, "reranked_results": vec_top})


async def ingest_document(
    doc_id: str,
    text: str,
    *,
    section_id: str | None = None,
) -> dict[str, Any]:
    if not doc_id.strip():
        raise ValueError("doc_id is required")
    chunks = await llm_service.chunk_document(text)
    sid = _resolve_section_id_for_ingest(doc_id, section_id)
    if not chunks:
        return {
            "inserted": 0,
            "doc_id": doc_id,
            "section_id": sid,
            "collection": _collection_name,
        }

    embeddings = await embed_texts(chunks)
    if embeddings is None:
        raise RuntimeError("Embedding service returned no vectors; check OPENAI_BASE_URL and EMBEDDING_MODEL.")

    try:
        async with _lock:
            inserted = await asyncio.to_thread(_insert_sync, doc_id, chunks, embeddings, sid)
    except Exception as e:
        logger.exception("Milvus ingest failed for doc_id=%s", doc_id)
        raise MilvusStoreError(str(e)) from e
    return {
        "inserted": inserted,
        "doc_id": doc_id,
        "section_id": sid,
        "collection": _collection_name,
    }


async def search(
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
    *,
    hybrid: bool = True,
    rrf_k: int = 60,
    rrf_k_bm25: int | None = None,
    cross_encoder: bool = True,
    rerank_pool: int = 40,
    candidate_multiplier: int = 8,
    include_timings: bool = False,
) -> dict[str, Any]:
    q = query.strip()
    if not q:
        out: dict[str, Any] = {"hybrid_results": [], "reranked_results": []}
        if include_timings:
            out["timings_ms"] = {
                "query_embedding_ms": 0.0,
                "retrieval_thread_ms": 0.0,
                **_empty_timings_ms(),
            }
        return out
    t_embed = time.perf_counter()
    emb_list = await embed_texts([q])
    query_embedding_ms = (time.perf_counter() - t_embed) * 1000.0
    if emb_list is None or not emb_list:
        raise RuntimeError("Could not embed query text.")
    vec = emb_list[0]
    try:
        async with _lock:
            t_thread = time.perf_counter()
            result = await asyncio.to_thread(
                _search_sync,
                vec,
                q,
                top_k,
                doc_id,
                hybrid,
                rrf_k,
                rrf_k_bm25,
                cross_encoder,
                rerank_pool,
                candidate_multiplier,
                collect_timings=include_timings,
            )
            retrieval_thread_ms = (time.perf_counter() - t_thread) * 1000.0
        if include_timings:
            result = dict(result)
            tm = dict(result.pop("timings_ms", _empty_timings_ms()))
            tm["query_embedding_ms"] = query_embedding_ms
            tm["retrieval_thread_ms"] = retrieval_thread_ms
            result["timings_ms"] = tm
        return result
    except Exception as e:
        logger.exception("Milvus search failed")
        raise MilvusStoreError(str(e)) from e
