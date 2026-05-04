import logging
import os
from typing import Final

import httpx

logger = logging.getLogger(__name__)

# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
EMBEDDING_MODEL: Final[str] = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBEDDING_DIM: Final[int] = int(os.getenv("EMBEDDING_DIMENSION", "768"))
BATCH_SIZE: Final[int] = 64


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Call OpenAI-compatible `/v1/embeddings`. Returns None on failure."""
    if not texts:
        return []
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "lm-studio")
    url = f"{base}/embeddings"
    out: list[list[float]] = []
    timeout = httpx.Timeout(60.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                r = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": EMBEDDING_MODEL, "input": batch},
                )
                r.raise_for_status()
                data = r.json().get("data") or []
                by_index = {item["index"]: item["embedding"] for item in data if "embedding" in item}
                for j in range(len(batch)):
                    emb = by_index.get(j)
                    if emb is None:
                        logger.warning("Missing embedding for index %s", i + j)
                        return None
                    if len(emb) != EMBEDDING_DIM:
                        logger.warning(
                            "Embedding dim %s != expected %s (set EMBEDDING_DIMENSION)",
                            len(emb),
                            EMBEDDING_DIM,
                        )
                        return None
                    out.append(emb)
        return out
    except Exception as e:
        logger.warning("Embedding request failed: %s", e)
        return None
