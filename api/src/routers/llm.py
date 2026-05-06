import time

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile

from ..dto.llm.index_document import IndexDocumentDto
from ..dto.llm.query import QueryDto
from ..dto.llm.search import SearchDto
from ..entities.llm.query_response import QueryMetrics, QueryResponse
from ..entities.llm.search_response import SearchResponse
from ..services import evaluation_service, llm_service, vector_store_service
from ..services.vector_store_service import MilvusStoreError

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/index")
async def index_document(body: IndexDocumentDto) -> dict:
    try:
        return await vector_store_service.ingest_document(
            body.doc_id.strip(),
            body.document,
            section_id=(body.section_id or "").strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except MilvusStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/index-file")
async def index_document_file(
    doc_id: str = Query(..., description="Stable id for this document."),
    section_id: str | None = Query(
        default=None,
        max_length=256,
        description="Section id per chunk; if omitted or blank, defaults to doc_id.",
    ),
    file: UploadFile = File(...),
) -> dict:
    data = await file.read()
    try:
        text = await llm_service.extract_text(data, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return await vector_store_service.ingest_document(
            doc_id.strip(),
            text,
            section_id=(section_id or "").strip(),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except MilvusStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/index-binary")
async def index_document_binary(
    request: Request,
    doc_id: str = Query(..., description="Stable id for this document."),
    section_id: str | None = Query(
        default=None,
        max_length=256,
        description="Section id per chunk; if omitted or blank, defaults to doc_id.",
    ),
    filename: str | None = Query(
        default=None,
        description="Original filename for type detection (e.g. report.pdf).",
    ),
    x_filename: str | None = Header(default=None, alias="X-Filename"),
) -> dict:
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body")
    name = (filename or x_filename or "").strip()
    try:
        text = await llm_service.extract_text(
            data,
            filename=name or None,
            content_type=request.headers.get("content-type"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return await vector_store_service.ingest_document(
            doc_id.strip(),
            text,
            section_id=(section_id or "").strip(),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except MilvusStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/search", response_model=SearchResponse)
async def search_chunks(body: SearchDto) -> SearchResponse:
    try:
        out = await vector_store_service.search(
            body.query,
            top_k=body.top_k,
            doc_id=body.doc_id,
            hybrid=body.hybrid,
            rrf_k=body.rrf_k,
            rrf_k_bm25=body.rrf_k_bm25,
            cross_encoder=body.cross_encoder,
            rerank_pool=body.rerank_pool,
            candidate_multiplier=body.candidate_multiplier,
        )
        return SearchResponse(**out)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except MilvusStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/query", response_model=QueryResponse)
async def query_chunks(body: QueryDto) -> QueryResponse:
    try:
        t_handler = time.perf_counter()
        latencies_ms: dict[str, float] = {}
        retrieved = await vector_store_service.search(
            body.query,
            top_k=body.top_k,
            doc_id=body.doc_id,
            hybrid=body.hybrid,
            rrf_k=body.rrf_k,
            rrf_k_bm25=body.rrf_k_bm25,
            cross_encoder=body.cross_encoder,
            rerank_pool=body.rerank_pool,
            candidate_multiplier=body.candidate_multiplier,
            include_timings=body.metrics,
        )
        if body.metrics:
            latencies_ms.update(dict(retrieved.get("timings_ms") or {}))
        context_results = retrieved.get("reranked_results") or []
        answer, gen_latencies = await llm_service.generate_answer(
            body.query,
            context_results,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            collect_timings=body.metrics,
        )
        if body.metrics:
            latencies_ms.update(gen_latencies)
        metrics: QueryMetrics | None = None
        if body.metrics:
            raw = evaluation_service.build_query_metrics(
                hybrid_results=retrieved.get("hybrid_results") or [],
                reranked_results=context_results,
                k=body.top_k,
                answer=answer,
                latencies_ms=latencies_ms,
            )
            extra = await evaluation_service.compute_additional_ragas_metrics(
                user_input=body.query,
                response=answer,
                reranked_results=context_results,
            )
            ar = extra.get("answer_relevancy")
            if ar is not None:
                raw["answer_relevancy"] = ar
                dm = ar.get("duration_ms")
                if isinstance(dm, (int, float)):
                    latencies_ms["metrics_answer_relevancy_ms"] = float(dm)
            cr = extra.get("context_relevance")
            if cr is not None:
                raw["context_relevance"] = cr
                dm = cr.get("duration_ms")
                if isinstance(dm, (int, float)):
                    latencies_ms["metrics_context_relevance_ms"] = float(dm)
            fb = await evaluation_service.compute_faithfulness_ragas(
                user_input=body.query,
                response=answer,
                reranked_results=context_results,
            )
            if fb is not None:
                raw["faithfulness"] = fb
                dm = fb.get("duration_ms")
                if isinstance(dm, (int, float)):
                    latencies_ms["metrics_faithfulness_ms"] = float(dm)
            latencies_ms["query_handler_wall_ms"] = (time.perf_counter() - t_handler) * 1000.0
            raw["latencies_ms"] = evaluation_service.floor_latency_map_ms(dict(latencies_ms))
            metrics = QueryMetrics.model_validate(raw)
        return QueryResponse(
            answer=answer,
            context_results=context_results,
            metrics=metrics,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except MilvusStoreError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e