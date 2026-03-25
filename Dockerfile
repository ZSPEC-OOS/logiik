FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cognita/ ./cognita/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Create knowledge_base directory (can be overridden by volume mount)
RUN mkdir -p knowledge_base/{embeddings,checkpoints,training_data,metadata}

# Expose ports
EXPOSE 8000 8501

# Copy and configure entrypoint
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
