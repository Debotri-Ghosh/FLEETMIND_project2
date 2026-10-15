# ── Stage 1: build wheel ─────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml constraints.txt ./
COPY fleetmind ./fleetmind
COPY alembic ./alembic
COPY alembic.ini ./
RUN pip install --no-cache-dir --prefix=/install . -c constraints.txt

# ── Stage 2: runtime ─────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY alembic ./alembic
COPY alembic.ini ./
EXPOSE 8000
CMD ["uvicorn", "fleetmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
