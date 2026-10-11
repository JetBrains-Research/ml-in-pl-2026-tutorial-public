# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.23 AS uv
FROM python:3.13.14-slim-bookworm

COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra openenv --no-install-project

ENV HOME=/tmp

LABEL org.opencontainers.image.source="https://github.com/JetBrains-Research/ml-in-pl-2026-tutorial-public" \
    org.opencontainers.image.title="ML in PL Module 2: NeMo Gym" \
    org.opencontainers.image.description="Four NeMo Gym evaluations with IDEGym sandbox verification"
ENTRYPOINT ["/app/.venv/bin/python", "-m", "experiments"]
CMD ["--help"]

# Source-only edits rebuild this layer while reusing the locked dependencies.
COPY . .
