FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV FORESIGHTX_ARTIFACTS_DIR=/app/artifacts/model

WORKDIR /app

COPY ForesightX-Pattern/requirements.inference.txt /app/requirements.inference.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.inference.txt

COPY ForesightX-Pattern/configs/default.yaml /app/configs/default.yaml
COPY ForesightX-Pattern/artifacts/model/model.onnx /app/artifacts/model/model.onnx
COPY ForesightX-Pattern/artifacts/model/scaler.pkl /app/artifacts/model/scaler.pkl
COPY ForesightX-Pattern/artifacts/model/metadata.json /app/artifacts/model/metadata.json
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
