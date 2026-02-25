# ── Stage 1: dependency layer (cached by Docker unless requirements.txt changes) ──
FROM python:3.11-slim AS deps

WORKDIR /app

# Install system libs needed by soundfile / pydub / numpy / ffmpeg (for audio convert)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first — Docker caches this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: app layer (rebuilt on every code push — fast, just copying files) ──
FROM deps AS app

WORKDIR /app

# Copy source code, data, and pre-built chroma_db
COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# Create logs dir so gunicorn doesn't fail on first start
RUN mkdir -p logs/audio logs/metrics

ENV PYTHONUNBUFFERED=1
# Skip heavy vector index build at startup on free tier (keyword search still works)
ENV SKIP_VECTOR_INDEX=1

EXPOSE 8000

# Run the Flask web_call_app with gunicorn (1 worker + 6 threads for concurrent calls)
CMD ["gunicorn", "-w", "1", "--threads", "6", "-b", "0.0.0.0:8000", "--timeout", "120", "--chdir", "src", "web_call_app:app"]
