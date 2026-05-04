import base64
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from .prisma import prisma
from .routers import llm, user


def _sanitize_validation_detail(obj: object) -> object:
    if isinstance(obj, bytes):
        preview = base64.b64encode(obj[:80]).decode("ascii")
        return f"<binary {len(obj)} bytes, base64 prefix: {preview}...>"
    if isinstance(obj, dict):
        return {str(k): _sanitize_validation_detail(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_validation_detail(v) for v in obj]
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    await prisma.connect()
    yield
    print("Shutting down...")
    await prisma.disconnect()
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


