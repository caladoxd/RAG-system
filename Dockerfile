FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy schema and requirements early
COPY db/prisma ./db/prisma
COPY api/src/requirements.txt ./api/src/requirements.txt

# Install Python dependencies (including uvicorn)
RUN pip install --no-cache-dir -r ./api/src/requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24"

# Install prisma CLI and generate client
RUN pip install --no-cache-dir prisma \
    && prisma generate --schema=./db/prisma/schema.prisma

# Copy entire api directory to ensure all source files are present
COPY api ./api

EXPOSE 8000

# Set PYTHONPATH to include /app so 'src.main:app' resolves correctly
ENV PYTHONPATH=/app/api

# Run uvicorn with correct module path relative to PYTHONPATH
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
