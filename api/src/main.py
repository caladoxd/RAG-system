import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from .prisma import prisma
from .routers import llm, user

logger = logging.getLogger(__name__)


def _sanitize_validation_detail(obj: object) -> object:
    if isinstance(obj, bytes):
        preview = base64.b64encode(obj[:80]).decode("ascii")
        return f"<binary {len(obj)} bytes, base64 prefix: {preview}...>"
    if isinstance(obj, dict):
        return {str(k): _sanitize_validation_detail(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_validation_detail(v) for v in obj]
    return obj


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


async def _run_startup_warmup() -> None:
    """Pay one-time ML / network costs before the first user request."""
    from .services import embedding_service, reranker_service, vector_store_service

    t0 = time.perf_counter()
    try:
        await asyncio.to_thread(reranker_service.warm_up_cross_encoder)
    except Exception:
        logger.exception("Cross-encoder startup warm-up failed")
    try:
        await asyncio.to_thread(vector_store_service.warm_up_milvus_sync)
    except Exception:
        logger.exception("Milvus startup warm-up failed")
    try:
        emb = await embedding_service.embed_texts(["warmup"])
        if emb is None:
            logger.warning(
                "Embedding warm-up returned no vectors; first retrieval may be slower"
            )
    except Exception:
        logger.exception("Embedding startup warm-up failed")
    logger.info("Startup warm-up finished in %.2fs", time.perf_counter() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    prisma_connected = False
    if _env_flag("SKIP_PRISMA_CONNECT"):
        logger.warning(
            "SKIP_PRISMA_CONNECT is set; skipping Prisma. User routes need a DB connection."
        )
    else:
        try:
            await prisma.connect()
            prisma_connected = True
        except Exception:
            logger.exception("Prisma connect failed")
            if _env_flag("PRISMA_FAIL_OPEN"):
                logger.warning(
                    "PRISMA_FAIL_OPEN is set; starting without database (user routes will error)."
                )
            else:
                raise
    if not _env_flag("SKIP_STARTUP_WARMUP"):
        await _run_startup_warmup()
    else:
        logger.warning("SKIP_STARTUP_WARMUP is set; skipping cross-encoder / Milvus / embedding warm-up")
    yield
    logger.info("Shutting down...")
    if prisma_connected:
        try:
            await prisma.disconnect()
        except Exception:
            logger.exception("Prisma disconnect failed")
    try:
        from pymilvus import connections

        if connections.has_connection("default"):
            connections.disconnect("default")
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_validation_detail(exc.errors())},
    )


@app.get('/health')
async def health():
    return {'status': 'ok'}

app.include_router(user.router)
app.include_router(llm.router)


