from typing import Any

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from ..services import llm_service, vector_store_service
from ..services.vector_store_service import MilvusStoreError

router = APIRouter(prefix="/llm", tags=["llm"])


class ChunkDocumentBody(BaseModel):
    document: str = Field(..., description="Plain text to chunk")


class IndexDocumentBody(BaseModel):
    doc_id: str = Field(..., description="Stable id for this document (re-index replaces prior chunks).")
    document: str = Field(..., description="Plain text to chunk, embed, and upsert into Milvus.")
    section_id: str | None = Field(
        default=None,
        max_length=256,
        description="Section / chapter id on every chunk; if omitted or blank, defaults to doc_id.",
    )


class SearchBody(BaseModel):
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


class SearchResponse(BaseModel):
    hybrid_results: list[dict[str, Any]] = Field(
        ...,
        description="After vector retrieval: BM25+RRF-ranked full candidate list when hybrid; else all vector hits ordered.",
    )
    reranked_results: list[dict[str, Any]] = Field(
        ...,
        description="Final top_k: cross-encoder order when enabled; else top RRF (or vector) hits.",
    )


@router.post("/chunk")
async def chunk_document(body: ChunkDocumentBody) -> list[str]:
    return await llm_service.chunk_document(body.document)


@router.post("/chunk-file")
async def chunk_document_file(file: UploadFile = File(...)) -> list[str]:
    data = await file.read()
    try:
        text = await llm_service.extract_text(data, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return await llm_service.chunk_document(text)


@router.post("/chunk-binary")
async def chunk_document_binary(
    request: Request,
    filename: str | None = Query(
        default=None,
        description="Original filename for type detection (e.g. notes.txt). Optional if body is PDF/DOCX magic bytes, UTF-8 plaintext, or Content-Type is text/*.",
    ),
    x_filename: str | None = Header(
        default=None,
        alias="X-Filename",
        description="Same as filename query param (e.g. notes.txt for raw binary uploads in Postman).",
    ),
) -> list[str]:
    """Body = raw file bytes (Postman: Body → binary). Not multipart."""
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
    return await llm_service.chunk_document(text)


@router.post("/index")
async def index_document(body: IndexDocumentBody) -> dict:
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
async def search_chunks(body: SearchBody) -> SearchResponse:
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