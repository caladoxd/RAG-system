import logging
import math
import os
import re
import time
from typing import Any, cast

from openai import AsyncOpenAI

try:
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness
except ImportError:  # pragma: no cover — minimal installs without RAGAS
    embedding_factory = None  # type: ignore[assignment,misc]
    llm_factory = None  # type: ignore[assignment,misc]
    AnswerRelevancy = None  # type: ignore[assignment,misc]
    ContextRelevance = None  # type: ignore[assignment,misc]
    Faithfulness = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def floor_latency_map_ms(values: dict[str, Any]) -> dict[str, int]:
    """Floor millisecond timings to whole milliseconds for stable JSON/API output."""
    out: dict[str, int] = {}
    for key, raw in values.items():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            out[key] = int(math.floor(float(raw)))
    return out


def compact_faithfulness_error(message: str, max_len: int = 480) -> str:
    """Shorten noisy RAGAS / Instructor multi-attempt traces for API consumers."""
    if not message:
        return ""
    if "multiple values for keyword argument 'mode'" in message:
        return (
            "RAGAS/Instructor duplicate `mode`: the judge stack already passes mode internally—"
            "do not pass Instructor mode into llm_factory. Retry the request; for LM Studio try "
            "RAGAS_LLM_ADAPTER=litellm (install litellm) or a cloud judge."
        )[:max_len]
    if "agenerate()" in message.lower() and "synchronous client" in message.lower():
        return (
            "RAGAS Faithfulness needs an async OpenAI client: use AsyncOpenAI with "
            "llm_factory(model, client=async_client) and await ascore() (RAGAS score() "
            "still calls ascore internally)."
        )[:max_len]
    if "response_format" in message or "failed_attempts" in message or "<generation" in message:
        hint = (
            "RAGAS judge used a response_format this server does not accept. "
            "For LM Studio / local OpenAI-compatible APIs try RAGAS_LLM_ADAPTER=litellm, "
            "or use an API that supports json_schema. "
        )
        m = re.search(r"Error code:\s*\d+\s*[-—]\s*[^\n<]{0,220}", message)
        tail = (m.group(0).strip() + ".") if m else ""
        combined = (hint + tail).strip()
        return combined[:max_len]
    return message[:max_len]


def floor_ms(value: float | int) -> int:
    return int(math.floor(float(value)))


def _is_response_format_error(message: str) -> bool:
    m = message.lower()
    return "response_format" in m and "json_schema" in m


def _is_sync_client_agenerate_error(exc: BaseException) -> bool:
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(parts) < 8:
        oid = id(cur)
        if oid in seen:
            break
        seen.add(oid)
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    blob = " ".join(parts).lower()
    return "agenerate()" in blob and "synchronous client" in blob


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


def _retrieved_context_strings(
    reranked_results: list[dict[str, Any]],
    *,
    top_n: int = 5,
    max_chars_per_chunk: int = 1400,
) -> list[str]:
    """Prepare cleaner, bounded contexts for LLM judges."""
    texts: list[str] = []
    for row in reranked_results[: max(1, top_n)]:
        t = str(row.get("text") or "").strip()
        if t:
            # PDF extraction often includes excessive whitespace/newlines.
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) > max_chars_per_chunk:
                t = t[:max_chars_per_chunk].rstrip() + "..."
            texts.append(t)
    return texts


def _llm_metric_payload(metric: str, score: float | None, error: str | None, duration_ms: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "score": score,
        "error": error,
        "duration_ms": floor_ms(duration_ms),
    }


def _coerce_score_0_1(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return max(0.0, min(1.0, score))


def _parse_zero_one_score(text: str) -> float | None:
    """Parse a score in [0, 1] from model output.

    Heuristic order:
    1) Prefer explicit labels like "score: 0.82"
    2) Prefer a standalone numeric last line
    3) Fallback to the last valid numeric token in the full text
    """
    raw = (text or "").strip()
    if not raw:
        return None

    explicit = re.search(
        r"(?i)\b(?:score|faithfulness|relevancy|relevance)\b\s*[:=]\s*(0?\.\d+|0|1(?:\.0+)?)\b",
        raw,
    )
    if explicit:
        try:
            return max(0.0, min(1.0, float(explicit.group(1))))
        except ValueError:
            pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if re.fullmatch(r"(0?\.\d+|0|1(?:\.0+)?)", last):
            try:
                return max(0.0, min(1.0, float(last)))
            except ValueError:
                pass

    matches = re.findall(r"\b(0?\.\d+|0|1(?:\.0+)?)\b", raw)
    if not matches:
        return None
    try:
        v = float(matches[-1])
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


async def _faithfulness_text_judge(
    *,
    user_input: str,
    response: str,
    contexts: list[str],
    base: str,
    key: str,
    model: str,
    max_ctx_chars: int = 1400,
) -> float:
    """Plain chat.completions (no response_format) for OpenAI-compatible servers that reject JSON mode."""
    lines: list[str] = []
    for i, c in enumerate(contexts):
        piece = c[:max_ctx_chars] + ("…" if len(c) > max_ctx_chars else "")
        lines.append(f"[{i + 1}] {piece}")
    ctx_block = "\n".join(lines)
    prompt = (
        "You judge whether an assistant answer is faithful to the reference context only.\n"
        "Ignore your own knowledge; use only the numbered snippets.\n\n"
        f"User question:\n{user_input}\n\n"
        f"Assistant answer:\n{response}\n\n"
        f"Reference context:\n{ctx_block}\n\n"
        "Output exactly one decimal number from 0 to 1 inclusive (faithfulness). "
        "1 means every substantive claim is supported; 0 means unsupported or contradicted. "
        "No words, no punctuation besides the number."
    )
    client = AsyncOpenAI(api_key=key, base_url=base)
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32,
    )
    content = (r.choices[0].message.content or "").strip()
    score = _parse_zero_one_score(content)
    if score is None:
        raise ValueError(f"judge returned no 0-1 score: {content[:160]!r}")
    return score


async def compute_faithfulness_ragas(
    *,
    user_input: str,
    response: str,
    reranked_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """RAGAS faithfulness: claims in the answer vs retrieved contexts (requires LLM judge)."""
    contexts = _retrieved_context_strings(
        reranked_results,
        top_n=5,
        max_chars_per_chunk=1800,
    )
    if not contexts or not (response or "").strip():
        return None
    t_start = time.perf_counter()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - t_start) * 1000.0

    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "lm-studio")
    model = (
        (os.getenv("RAGAS_EVAL_MODEL") or "").strip()
        or (os.getenv("GENERATION_MODEL") or "").strip()
        or "gpt-4o-mini"
    )

    if llm_factory is None or Faithfulness is None:
        logger.info("RAGAS not installed; using text-mode faithfulness judge only")
        try:
            score = await _faithfulness_text_judge(
                user_input=user_input,
                response=response,
                contexts=contexts,
                base=base,
                key=key,
                model=model,
            )
            return {
                "metric": "faithfulness",
                "score": score,
                "error": None,
                "duration_ms": floor_ms(_elapsed_ms()),
            }
        except Exception as fb_e:
            logger.warning("Faithfulness failed (text judge, no RAGAS): %s", fb_e)
            return {
                "metric": "faithfulness",
                "score": None,
                "error": compact_faithfulness_error(str(fb_e)),
                "duration_ms": floor_ms(_elapsed_ms()),
            }

    ragas_llm_factory = llm_factory
    RagasFaithfulness = Faithfulness
    assert ragas_llm_factory is not None and RagasFaithfulness is not None

    async_client = AsyncOpenAI(api_key=key, base_url=base)

    async def _ascore_with(llm_extra: dict[str, Any]) -> Any:
        """RAGAS collections Faithfulness only supports async scoring via ascore()."""
        llm = ragas_llm_factory(model, client=async_client, **llm_extra)
        return await RagasFaithfulness(llm=llm).ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=contexts,
        )

    try:
        explicit = os.getenv("RAGAS_LLM_ADAPTER", "").strip().lower()
        if explicit in ("litellm", "instructor"):
            result = await _ascore_with({"adapter": explicit})
        elif explicit in ("none", "minimal", "default"):
            result = await _ascore_with({})
        else:
            last: BaseException | None = None
            result: Any = None
            # LM Studio often rejects Instructor JSON mode; try litellm before default auto.
            for llm_extra in (
                {"adapter": "litellm"},
                {},
                {"adapter": "instructor"},
            ):
                try:
                    result = await _ascore_with(llm_extra)
                    break
                except Exception as e:
                    last = e
                    if _is_response_format_error(str(e)) or _is_sync_client_agenerate_error(
                        e
                    ):
                        continue
                    raise
            if result is None:
                raise last if last is not None else RuntimeError("faithfulness: no result")

        raw_val = getattr(result, "value", result)
        try:
            score = float(raw_val) if raw_val is not None else None
        except (TypeError, ValueError):
            score = None
        return {
            "metric": "faithfulness",
            "score": score,
            "error": None,
            "duration_ms": floor_ms(_elapsed_ms()),
        }
    except Exception as ragas_e:
        logger.info("RAGAS faithfulness failed, trying text-mode judge: %s", ragas_e)
        try:
            score = await _faithfulness_text_judge(
                user_input=user_input,
                response=response,
                contexts=contexts,
                base=base,
                key=key,
                model=model,
            )
            return {
                "metric": "faithfulness",
                "score": score,
                "error": None,
                "duration_ms": floor_ms(_elapsed_ms()),
            }
        except Exception as fb_e:
            logger.warning("Faithfulness failed (RAGAS then text judge): %s | %s", ragas_e, fb_e)
            return {
                "metric": "faithfulness",
                "score": None,
                "error": compact_faithfulness_error(f"{ragas_e!s}; fallback: {fb_e!s}"),
                "duration_ms": floor_ms(_elapsed_ms()),
            }


async def compute_additional_ragas_metrics(
    *,
    user_input: str,
    response: str,
    reranked_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute AnswerRelevancy + ContextRelevance with isolated failures per metric."""
    contexts = _retrieved_context_strings(
        reranked_results,
        top_n=1,
        max_chars_per_chunk=1200,
    )
    if not contexts or not (response or "").strip() or not (user_input or "").strip():
        return {}

    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "lm-studio")
    model = (
        (os.getenv("RAGAS_EVAL_MODEL") or "").strip()
        or (os.getenv("GENERATION_MODEL") or "").strip()
        or "gpt-4o-mini"
    )
    emb_model = (os.getenv("RAGAS_EMBEDDING_MODEL") or "").strip() or "text-embedding-3-small"

    if (
        llm_factory is None
        or embedding_factory is None
        or AnswerRelevancy is None
        or ContextRelevance is None
    ):
        msg = "RAGAS dependencies missing for additional metrics"
        return {
            "answer_relevancy": _llm_metric_payload(
                "answer_relevancy", None, compact_faithfulness_error(msg), 0.0
            ),
            "context_relevance": _llm_metric_payload(
                "context_relevance", None, compact_faithfulness_error(msg), 0.0
            ),
        }

    ragas_llm_factory = llm_factory
    ragas_embedding_factory = embedding_factory
    RagasAnswerRelevancy = AnswerRelevancy
    RagasContextRelevance = ContextRelevance
    assert ragas_llm_factory is not None
    assert ragas_embedding_factory is not None
    assert RagasAnswerRelevancy is not None
    assert RagasContextRelevance is not None

    async_client = AsyncOpenAI(api_key=key, base_url=base)
    explicit = os.getenv("RAGAS_LLM_ADAPTER", "").strip().lower()
    if explicit in ("litellm", "instructor"):
        llm_attempts: list[dict[str, Any]] = [{"adapter": explicit}]
    elif explicit in ("none", "minimal", "default"):
        llm_attempts = [{}]
    else:
        # Local OpenAI-compatible servers are usually happier with litellm first.
        llm_attempts = [{"adapter": "litellm"}, {}, {"adapter": "instructor"}]

    out: dict[str, dict[str, Any]] = {}

    async def _answer_relevancy_text_fallback() -> float:
        prompt = (
            "Score how relevant the assistant answer is to the user question.\n"
            "Return only a number between 0 and 1.\n\n"
            f"User question:\n{user_input}\n\n"
            f"Assistant answer:\n{response}\n\n"
            "0 means off-topic or evasive. 1 means directly and completely answers the question."
        )
        r = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )
        s = _coerce_score_0_1(_parse_zero_one_score(r.choices[0].message.content or ""))
        if s is None:
            raise ValueError("answer_relevancy fallback judge did not return a valid 0-1 score")
        return s

    async def _context_relevance_text_fallback() -> float:
        ctx = "\n".join(f"[{i + 1}] {c[:1200]}" for i, c in enumerate(contexts))
        prompt = (
            "Score how relevant the retrieved context is to the user question.\n"
            "Return only a number between 0 and 1.\n\n"
            f"User question:\n{user_input}\n\n"
            f"Retrieved context:\n{ctx}\n\n"
            "0 means irrelevant. 1 means highly relevant and sufficient."
        )
        r = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )
        s = _coerce_score_0_1(_parse_zero_one_score(r.choices[0].message.content or ""))
        if s is None:
            raise ValueError("context_relevance fallback judge did not return a valid 0-1 score")
        return s

    t0 = time.perf_counter()
    try:
        result: Any = None
        last: Exception | None = None
        for llm_extra in llm_attempts:
            try:
                llm = ragas_llm_factory(model, client=async_client, **llm_extra)
                embeddings = ragas_embedding_factory(
                    provider="openai",
                    model=emb_model,
                    client=async_client,
                )
                metric = RagasAnswerRelevancy(llm=llm, embeddings=cast(Any, embeddings))
                result = await metric.ascore(user_input=user_input, response=response)
                break
            except Exception as e:
                last = e
                if _is_response_format_error(str(e)) or _is_sync_client_agenerate_error(e):
                    continue
                raise
        if result is None:
            raise last if last is not None else RuntimeError("answer_relevancy: no result")
        raw = getattr(result, "value", result)
        score = _coerce_score_0_1(raw)
        if score is None:
            raise ValueError("answer_relevancy returned NaN/invalid score")
        out["answer_relevancy"] = _llm_metric_payload(
            "answer_relevancy", score, None, (time.perf_counter() - t0) * 1000.0
        )
    except Exception as e:
        try:
            score = await _answer_relevancy_text_fallback()
            out["answer_relevancy"] = _llm_metric_payload(
                "answer_relevancy", score, None, (time.perf_counter() - t0) * 1000.0
            )
        except Exception as fb_e:
            err = compact_faithfulness_error(f"{e}; fallback: {fb_e}")
            out["answer_relevancy"] = _llm_metric_payload(
                "answer_relevancy",
                None,
                err,
                (time.perf_counter() - t0) * 1000.0,
            )

    t1 = time.perf_counter()
    try:
        best_score: float | None = None
        last: Exception | None = None
        any_success = False
        for llm_extra in llm_attempts:
            try:
                llm = ragas_llm_factory(model, client=async_client, **llm_extra)
                metric = RagasContextRelevance(llm=llm)
                result = await metric.ascore(user_input=user_input, retrieved_contexts=contexts)
                any_success = True
                raw = getattr(result, "value", result)
                score_try = _coerce_score_0_1(raw)
                if score_try is not None:
                    if best_score is None or score_try > best_score:
                        best_score = score_try
            except Exception as e:
                last = e
                if _is_response_format_error(str(e)) or _is_sync_client_agenerate_error(e):
                    continue
                raise
        if not any_success:
            raise last if last is not None else RuntimeError("context_relevance: no result")
        score = best_score
        if score is None:
            raise ValueError("context_relevance returned NaN/invalid score")
        out["context_relevance"] = _llm_metric_payload(
            "context_relevance", score, None, (time.perf_counter() - t1) * 1000.0
        )
    except Exception as e:
        try:
            score = await _context_relevance_text_fallback()
            out["context_relevance"] = _llm_metric_payload(
                "context_relevance", score, None, (time.perf_counter() - t1) * 1000.0
            )
        except Exception as fb_e:
            err = compact_faithfulness_error(f"{e}; fallback: {fb_e}")
            out["context_relevance"] = _llm_metric_payload(
                "context_relevance",
                None,
                err,
                (time.perf_counter() - t1) * 1000.0,
            )

    return out


def build_query_metrics(
    *,
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    k: int,
    answer: str,
    latencies_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    rr = retrieval_recall_at_k(hybrid_results, reranked_results, k)
    if latencies_ms is not None:
        latencies_ms["metrics_retrieval_recall_at_k_ms"] = float(
            (time.perf_counter() - t0) * 1000.0
        )
    if rr is not None:
        out["retrieval_recall_at_k"] = rr
    t1 = time.perf_counter()
    cc = citation_coverage(answer, len(reranked_results))
    if latencies_ms is not None:
        latencies_ms["metrics_citations_ms"] = float((time.perf_counter() - t1) * 1000.0)
    if cc is not None:
        out["citations"] = cc
    return out
