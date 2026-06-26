FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY api/src/requirements.txt ./requirements.txt
# Deployer runs the app with uvicorn — install even if the repo only lists fastapi.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24"

COPY db ./db
COPY api/src ./src

ENV PYTHONPATH=/app

# Generate Prisma client before starting the app.
RUN cd /app && python -m pip install prisma \
    && python -m prisma generate --schema=db/prisma/schema.prisma

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]