FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY db/prisma ./db/prisma
COPY api/src/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24" \
    && pip install --no-cache-dir prisma

COPY api/src ./src

ENV PYTHONPATH=/app

# Generate Prisma client at build time
RUN prisma generate --schema=./db/prisma/schema.prisma || true

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]