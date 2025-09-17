# Multi-stage build for serving multilingual-e5-small embedding model
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 
# Copy UV from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY pyproject.toml ./

# Create data directory and set permissions
RUN mkdir -p data/models && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
# RUN uv venv
# RUN uv sync 
# ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Download model during build
RUN uv run app/download_model.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD uv run python -c "import requests; requests.get('http://localhost:8000/health')"

# Start the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
