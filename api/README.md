# API

FastAPI application: users (Prisma/Postgres), LLM helpers (chunking, extract PDF/DOCX/TXT), embeddings, and **Milvus** ingest + hybrid (vector + BM25) search.

## Run

From this directory:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
```

Generate the Prisma client (schema lives under `../db/prisma/`):

```bash
cd ../db/prisma && ../../api/.venv/bin/python -m prisma generate --schema schema.prisma
```

Copy or edit `.env` (see comments there for `DB_URL`, Milvus, embeddings).

```bash
cd ../api
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for OpenAPI.

## Layout

- `main.py` — ASGI entry (`uvicorn main:app`).
- `src/` — application package (`main`, `routers`, `services`, generated `prisma_client/`).

## Related

- Database schema: `../db/`
- Docker stack (Milvus, Redis, …): `../compose.yml`
