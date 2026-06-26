FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY api/src/requirements.txt ./requirements.txt
# Deployer runs the app with uvicorn — install even if the repo only lists fastapi.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24"

# Install Node.js for Prisma CLI (required to generate Prisma client).
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY db ./db
COPY api/src ./src

# Generate Prisma client before starting the app.
# The schema is at db/prisma/schema.prisma; set PRISMA_CLI_BINARY_TARGETS for Linux.
RUN cd /app && npm install -g prisma --force \
    && PRISMA_CLI_BINARY_TARGETS=linux-musl prisma generate --schema=db/prisma/schema.prisma

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]