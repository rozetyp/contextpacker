# ContextPacker API - Production Dockerfile
# Railway auto-detects this file

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git (needed for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY context_packer/ ./context_packer/
COPY static/ ./static/

# Create directory for repo cache
RUN mkdir -p /app/repo_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway uses $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run with uvicorn
# Railway sets PORT env var, we default to 8000
CMD ["sh", "-c", "uvicorn context_packer.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
