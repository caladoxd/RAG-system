"""ASGI entrypoint: run `uvicorn main:app --reload` from this directory (`api/`)."""

from src.main import app

__all__ = ["app"]
