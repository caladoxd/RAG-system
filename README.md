# RAG System

Production-oriented Retrieval-Augmented Generation (RAG) application with document ingestion, hybrid retrieval, answer generation, and optional evaluation diagnostics.

## What It Does

- Indexes documents from:
  - plain text
  - uploaded files (`PDF`, `DOCX`, `TXT`)
  - raw binary uploads with content sniffing
- Chunks and embeds content into Milvus
- Runs hybrid retrieval:
  - ANN vector search
  - BM25 + RRF fusion
  - cross-encoder reranking
- Generates grounded answers with retrieved context
- Optionally computes evaluation diagnostics:
  - retrieval recall@k
  - citation coverage
  - faithfulness (RAGAS + robust fallback)
  - per-step latencies

## Tech Stack

### Backend

- Python, FastAPI
- Milvus (vector DB)
- BM25 (`rank-bm25`)
- Cross-encoder reranker (`sentence-transformers`)
- OpenAI-compatible APIs (`openai` SDK, local or cloud models)
- RAG evaluation (`ragas`, `litellm`)
- Prisma client (user module / DB integration)

### Frontend

- React + TypeScript + Vite
- Tailwind tooling + custom CSS

### Infra

- Docker Compose for Milvus dependencies (`etcd`, `minio`, `milvus-standalone`, `redis`)

## Repository Structure

```text
RAG-system/
├── api/          # FastAPI app (routers, services, dto, entities)
├── ui/           # React frontend
├── db/           # Prisma schema/migrations
└── compose.yml   # Local infra for Milvus stack
```

## System Flow

1. Ingest document (`/llm/index*`)
2. Extract text (`PDF/DOCX/TXT`)
3. Chunk + embed
4. Store chunks/vectors in Milvus
5. Query (`/llm/query`)
6. Retrieve (ANN + BM25/RRF + rerank)
7. Generate answer
8. (Optional) Compute quality metrics

## Key API Endpoints

- `POST /llm/index` - index plain text
- `POST /llm/index-file` - index uploaded file
- `POST /llm/index-binary` - index binary body
- `POST /llm/search` - retrieval-only diagnostics
- `POST /llm/query` - full RAG answer (+ optional metrics)
- `GET /health` - health check

## Run Locally

## 1) Start infrastructure

From repo root:

```bash
docker compose -f compose.yml up -d
```

## 2) Start API

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

Generate Prisma client:

```bash
cd ../db/prisma
../../api/.venv/bin/python -m prisma generate --schema schema.prisma
cd ../../api
```

Run backend:

```bash
uvicorn src.main:app --reload
```

Open API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 3) Start UI

```bash
cd ../ui
npm install
npm run dev
```

Open UI: [http://127.0.0.1:5173](http://127.0.0.1:5173)

## Environment Notes

Common variables used by the API:

- `OPENAI_BASE_URL` - OpenAI-compatible base URL (e.g., LM Studio or cloud)
- `OPENAI_API_KEY` - API key/token
- `GENERATION_MODEL` - model used for answer generation
- `RAGAS_EVAL_MODEL` - optional model for evaluation
- `MILVUS_HOST`, `MILVUS_PORT` - Milvus connection
- `SKIP_PRISMA_CONNECT=true` - run without DB on startup
- `PRISMA_FAIL_OPEN=true` - don't crash if Prisma connect fails
- `SKIP_STARTUP_WARMUP=true` - disable warm-up (faster boot, slower first query)

## Implementation Notes

- Practical handling of local-model quirks and OpenAI-compatible edge cases
- Startup warm-up to reduce cold-start latency
- Latency instrumentation for retrieval, generation, and metrics pipeline
- Robust evaluation path with graceful fallback when structured judge calls fail
- Cleaner domain modeling (`dto/` vs `entities/`) and modular service boundaries

## Future Improvements

- Add authentication and multi-tenant document namespaces
- Add regression/eval datasets and CI quality gates
- Add streaming responses in the UI
- Add observability dashboard for retrieval and model performance

