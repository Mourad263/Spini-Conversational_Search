# ---- Frontend build ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Python runtime (FastAPI serves API + built frontend) ----
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY schemas/ ./schemas/
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Demo checkout writes order JSON files here.
RUN mkdir -p /app/data/orders

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.routes:app --host 0.0.0.0 --port ${PORT:-8000}"]
