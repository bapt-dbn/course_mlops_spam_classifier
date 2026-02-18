FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --frozen \
    --no-dev \
    --no-install-project

FROM python:3.12-slim

RUN addgroup --system peepouser && \
    adduser --system peepouser && \
    adduser peepouser peepouser

WORKDIR /home/peepouser

COPY --chown=peepouser:peepouser --from=builder /app/.venv /home/peepouser/.venv

COPY --chown=peepouser:peepouser course_mlops/ ./course_mlops/
COPY --chown=peepouser:peepouser config/ ./config/
COPY --chown=peepouser:peepouser alembic.ini ./
COPY --chown=peepouser:peepouser migrations/ ./migrations/

ENV PATH="/home/peepouser/.venv/bin:$PATH" \
    PYTHONPATH=/home/peepouser \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER peepouser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=5s \
    --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')" || exit 1

ENTRYPOINT ["python", "-m", "course_mlops.cli"]
CMD ["--help"]
