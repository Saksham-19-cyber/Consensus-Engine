FROM python:3.12-slim

# Create a non-root user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and set ownership to user
COPY --chown=user:user . .

# Ensure data directories exist and are writable
RUN mkdir -p /app/data /app/data/sqlite /app/data/chroma && chown -R user:user /app/data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
