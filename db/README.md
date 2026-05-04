# Database

Prisma schema and tooling for **PostgreSQL** (used by `../api/`).

## Layout

- `prisma/schema.prisma` — models and `generator client` for **prisma-client-py** (output path points at `../api/src/prisma_client`).
- `requirements.txt` — Python deps if you run Prisma CLI from this folder’s venv.
- `.env` — typically `DATABASE_URL` / `DB_URL` for migrations and `prisma generate`.

## Generate client (after schema changes)

From `db/prisma` using the API project’s Python:

```bash
cd prisma
/path/to/api/.venv/bin/python -m prisma generate --schema schema.prisma
```

## Related

- API service that connects at startup: `../api/src/prisma.py`
- Compose / external Postgres: align `DB_URL` in `../api/.env` with your database host.
