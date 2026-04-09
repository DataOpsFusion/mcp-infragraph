# ── Stage 1: build deps ───────────────────────────────────────────────────────
# Explicit linux/amd64 so cross-compiling from Mac ARM works correctly
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /app

# Install build backend + deps first (layer-cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip hatchling && \
    (pip install --no-cache-dir ".[dev]" || pip install --no-cache-dir .)
# Copy source and install the package itself
COPY infragraph/ infragraph/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 infragraph

# Copy installed packages and source from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV MODE=sse \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chown -R infragraph:infragraph /app

USER infragraph

# Health check hits the SSE server ping endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

EXPOSE 8080

# Default: run MCP server in SSE/HTTP mode
CMD ["python", "-m", "infragraph.mcp.server"]
