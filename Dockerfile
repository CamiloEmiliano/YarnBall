# Dockerfile
FROM python:3.11-slim

# Install system libs needed for psycopg2 and kafka
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libpq-dev python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non‑root user
RUN useradd --create-home appuser
WORKDIR /app

USER appuser
RUN pip install --upgrade pip

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source
COPY . .

# Duplicate entrypoint lines removed
CMD ["tail","-f","/dev/null"]
