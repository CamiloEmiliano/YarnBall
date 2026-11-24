FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# System packages required for lightweight retrieval runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: Copy packaging metadata & requirements first to cache third-party layers
# (No in-container PyTorch or sentence-transformers; embeddings run via Ollama /api/embed per ADR-00035)
COPY pyproject.toml ./pyproject.toml
COPY retrieval/requirements.txt ./retrieval/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r retrieval/requirements.txt

# Stage 2: Selectively copy ONLY runtime source packages (no data/, tests/, docs/)
COPY retrieval ./retrieval
COPY knowledge_graph ./knowledge_graph
COPY embedding ./embedding
COPY ontology ./ontology
COPY contagion ./contagion
COPY synthesis ./synthesis
COPY curation ./curation
COPY sft_pipeline ./sft_pipeline
COPY universe ./universe
COPY tools ./tools

# Install local package cleanly without PYTHONPATH hacks
RUN pip install --no-cache-dir --no-deps .

# Build-time import smoke verification (ADR-00036 Invariant #5)
RUN python -c "import retrieval.service, retrieval.pipeline, synthesis.synthesizer"

EXPOSE 8000

CMD ["uvicorn", "retrieval.service:app", "--host", "0.0.0.0", "--port", "8000"]
