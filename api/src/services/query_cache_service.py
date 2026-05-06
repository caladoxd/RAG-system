import hashlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_client = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return _env_flag("QUERY_CACHE_ENABLED", True)


def _normalize_query(q: str) -> str:
    # Normalize punctuation and whitespace so near-identical phrasing shares keys.
    q = (q or "").lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


async def _get_client():
    global _client
    if _client is not None:
        return _client
    if not _enabled():
        return None
    try:
        from redis.asyncio import Redis

        host = os.getenv("REDIS_HOST", "127.0.0.1")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD") or None
        _client = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        await _client.ping()
        return _client
    except Exception as e:
        logger.warning("Query cache disabled (Redis unavailable): %s", e)
        return None


def _key_from_body(body: Any) -> str:
    payload = body.model_dump() if hasattr(body, "model_dump") else dict(body)
    payload["query"] = _normalize_query(str(payload.get("query") or ""))
    namespace = os.getenv("QUERY_CACHE_NAMESPACE", "rag-query-v1")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{h}"


async def get_cached_response(body: Any) -> dict[str, Any] | None:
    client = await _get_client()
    if client is None:
        return None
    key = _key_from_body(body)
    try:
        raw = await client.get(key)
    except Exception as e:
        logger.warning("Query cache read failed: %s", e)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def set_cached_response(body: Any, value: dict[str, Any]) -> None:
    client = await _get_client()
    if client is None:
        return
    key = _key_from_body(body)
    ttl = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "900"))
    try:
        await client.set(key, json.dumps(value, separators=(",", ":"), ensure_ascii=False), ex=ttl)
    except Exception as e:
        logger.warning("Query cache write failed: %s", e)
