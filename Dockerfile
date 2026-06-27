FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY db/prisma ./prisma
COPY api/src/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24" \
    && pip install --no-cache-dir prisma

COPY api/src ./src

ENV PYTHONPATH=/app

RUN cd /app && prisma generate --schema=/app/prisma/schema.prisma

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]