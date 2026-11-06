# Serving image. The champion model is scikit-learn, so TensorFlow stays out
# and the image is a fraction of the size a deep-learning image would be.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY kb ./kb
COPY models ./models

ENV JETWAY_LOG=/tmp/predictions.jsonl \
    PYTHONUNBUFFERED=1
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "uvicorn src.service:app --host 0.0.0.0 --port ${PORT:-8000}"]
