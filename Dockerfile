FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY db/prisma ./db/prisma
COPY api/src/requirements.txt ./api/src/requirements.txt

RUN pip install --no-cache-dir -r ./api/src/requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24" prisma

RUN prisma generate --schema=./db/prisma/schema.prisma

COPY api/src ./api/src

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
