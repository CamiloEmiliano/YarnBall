FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System packages required for Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev python3-dev libssl-dev libcurl4-openssl-dev make cmake openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (allows Docker cache reuse)
COPY rag/retriever/requirements.txt .

# Install Python dependencies (excluding pymgclient)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# pymgclient not needed – using neo4j driver


# Copy the application source code
COPY rag ./rag
COPY graph ./graph
COPY embedding ./embedding
COPY tools ./tools

EXPOSE 8000

CMD ["uvicorn", "rag.retriever.main:app", "--host", "0.0.0.0", "--port", "8000"]