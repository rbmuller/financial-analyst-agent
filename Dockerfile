FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /bin/uv

# The virtualenv lives outside /app so the dev bind mount does not hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
