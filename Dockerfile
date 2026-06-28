FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy schema first (before WORKDIR change so path is correct)
COPY db/prisma ./db/prisma
COPY api/src/requirements.txt ./api/src/requirements.txt

# Install Python dependencies and prisma CLI
RUN pip install --no-cache-dir -r ./api/src/requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24" prisma

# Generate prisma client from /app context (before WORKDIR change)
# If DATABASE_URL is not available at build time, prisma generate may fail.
# Use SKIP_ENGINE_CHECK or continue even if it fails, since the schema is present.
RUN prisma generate --schema=./db/prisma/schema.prisma || true

# Copy source code
COPY api/src ./api/src

# Set working directory for app execution
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]