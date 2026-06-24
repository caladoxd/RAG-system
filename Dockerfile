FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
# Deployer runs the app with uvicorn — install even if the repo only lists fastapi.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24"

COPY . .

# Prisma Python client (prisma_client) is generated — not committed to git.
# Schema path is repo-relative; COPY . . puts it at /app/<path>.
# The prisma-client-py query engine needs OpenSSL + CA certs at runtime,
# otherwise it fails to start with "Could not connect to the query engine".
RUN apt-get update && apt-get install -y --no-install-recommends openssl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir prisma \
    && prisma generate --schema=/app/db/prisma/schema.prisma

WORKDIR /app/api
ENV PYTHONPATH=/app/api

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
