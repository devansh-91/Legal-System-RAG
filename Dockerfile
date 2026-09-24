FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    CHROMA_DIR=/app/chroma_db

WORKDIR /app

# CPU-only PyTorch keeps the image small (sentence-transformers needs torch).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY legal_rag ./legal_rag
COPY web ./web
COPY data ./data

EXPOSE 8000
# Build the index on first start if it doesn't exist yet, then serve.
CMD ["sh", "-c", "[ -n \"$(ls -A \"$CHROMA_DIR\" 2>/dev/null)\" ] || python -m legal_rag.ingest; exec uvicorn legal_rag.api:app --host 0.0.0.0 --port 8000"]
