FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
# Deployer runs the app with uvicorn — install even if the repo only lists fastapi.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.24"

COPY . .


ENV PYTHONPATH=/app


EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
