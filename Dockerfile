FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ForesightX-Pattern/requirements.inference.txt /build/requirements.inference.txt
RUN pip install --no-cache-dir --prefix=/install -r /build/requirements.inference.txt

FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV FORESIGHTX_ARTIFACTS_DIR=/app/artifacts/model

WORKDIR /app

COPY --from=builder /install /usr/local

COPY ForesightX-Pattern/configs/default.yaml /app/configs/default.yaml
COPY ForesightX-Pattern/artifacts/model /app/artifacts/model
COPY ForesightX-Pattern/foresightx_pattern/__init__.py /app/foresightx_pattern/__init__.py
COPY ForesightX-Pattern/foresightx_pattern/app /app/foresightx_pattern/app
COPY ForesightX-Pattern/foresightx_pattern/ml/__init__.py /app/foresightx_pattern/ml/__init__.py
COPY ForesightX-Pattern/foresightx_pattern/ml/data/__init__.py /app/foresightx_pattern/ml/data/__init__.py
COPY ForesightX-Pattern/foresightx_pattern/ml/data/preprocessing.py /app/foresightx_pattern/ml/data/preprocessing.py
COPY ForesightX-Pattern/foresightx_pattern/ml/features /app/foresightx_pattern/ml/features
COPY ForesightX-Pattern/foresightx_pattern/ml/utils/__init__.py /app/foresightx_pattern/ml/utils/__init__.py
COPY ForesightX-Pattern/foresightx_pattern/ml/utils/config.py /app/foresightx_pattern/ml/utils/config.py
COPY ForesightX-Pattern/foresightx_pattern/ml/utils/markets.py /app/foresightx_pattern/ml/utils/markets.py

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=3).read()"

CMD ["uvicorn", "foresightx_pattern.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
